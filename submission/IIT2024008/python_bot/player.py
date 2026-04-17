
import eval7
from collections import Counter
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import STARTING_STACK, BIG_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot


class Player(Bot):
    def __init__(self):
        self.rank_map = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        self.range_cache = {}
        self.opp = {
            'showdowns': 0,
            'fold_seen': 0,
            'big_bets_seen': 0,
            'small_bets_seen': 0,
            'recent_deltas': []
        }
        self.round_id = 0
        self.decision_points = 0
        self.opp_bet_raise_events = 0
        self.opp_check_events = 0
        self.hand_opp_checks = 0
        self.we_bet_or_raised_this_hand = False
        self.our_bet_attempts = 0

    def handle_new_round(self, game_state, round_state, active):
        self.round_id = getattr(game_state, 'round_num', self.round_id + 1)
        self.hand_opp_checks = 0
        self.we_bet_or_raised_this_hand = False

    def handle_round_over(self, game_state, terminal_state, active):
        delta = terminal_state.deltas[active]
        self.opp['recent_deltas'].append(delta)
        if len(self.opp['recent_deltas']) > 20:
            self.opp['recent_deltas'].pop(0)

        prev = getattr(terminal_state, 'previous_state', None)
        if prev is None:
            return

        saw_showdown = False
        try:
            if FoldAction not in prev.legal_actions():
                saw_showdown = True
        except Exception:
            pass

        if saw_showdown:
            self.opp['showdowns'] += 1
        elif self.we_bet_or_raised_this_hand:
            self.opp['fold_seen'] += 1

    def get_range(self, s):
        if s not in self.range_cache:
            self.range_cache[s] = eval7.HandRange(s)
        return self.range_cache[s]

    def mix(self, my_cards, street, salt=0):
        key = ''.join(sorted(my_cards)) + '|' + str(self.round_id) + '|' + str(street) + '|' + str(salt)
        x = 0
        for i, ch in enumerate(key):
            x = (x * 131 + (i + 17) * ord(ch)) % 1000003
        return (x % 10000) / 10000.0

    def mark_aggr(self):
        self.we_bet_or_raised_this_hand = True
        self.our_bet_attempts += 1

    def pot_size(self, round_state):
        return (STARTING_STACK - round_state.stacks[0]) + (STARTING_STACK - round_state.stacks[1])

    def ranks_suits(self, cards):
        rs = [self.rank_map[c[0]] for c in cards]
        ss = [c[1] for c in cards]
        return rs, ss

    def chen_score(self, cards):
        r1 = self.rank_map[cards[0][0]]
        r2 = self.rank_map[cards[1][0]]
        hi, lo = max(r1, r2), min(r1, r2)
        suited = cards[0][1] == cards[1][1]
        pair = r1 == r2

        base_map = {14: 10, 13: 8, 12: 7, 11: 6, 10: 5}
        score = base_map.get(hi, hi / 2.0)

        if pair:
            score = max(5.0, score * 2.0)
            if hi == 5:
                score += 1.0
        if suited:
            score += 2.0

        gap = hi - lo - 1
        if gap == 1:
            score -= 1.0
        elif gap == 2:
            score -= 2.0
        elif gap == 3:
            score -= 4.0
        elif gap >= 4:
            score -= 5.0

        if gap <= 1 and hi < 12:
            score += 1.0
        return max(0.0, score)

    def hand_key(self, cards):
        r1 = cards[0][0]
        r2 = cards[1][0]
        v1 = self.rank_map[r1]
        v2 = self.rank_map[r2]
        if v1 == v2:
            return r1 + r2
        hi, lo = (r1, r2) if v1 > v2 else (r2, r1)
        suf = 's' if cards[0][1] == cards[1][1] else 'o'
        return hi + lo + suf

    def classify_preflop(self, cards):
        key = self.hand_key(cards)
        chen = self.chen_score(cards)
        premium = {'AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo', 'AQs'}
        strong = {'TT', '99', '88', 'AJs', 'ATs', 'AQo', 'KQs', 'KJs', 'QJs', 'JTs'}
        playable = {
            '77', '66', '55', '44', '33', '22',
            'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
            'KTs', 'QTs', 'J9s', 'T9s', '98s', '87s', '76s', '65s', '54s',
            'ATo', 'AJo', 'KQo', 'KJo', 'QJo', 'JTo'
        }
        if key in premium or chen >= 11.5:
            return 'premium', chen
        if key in strong or chen >= 9.0:
            return 'strong', chen
        if key in playable or chen >= 7.0:
            return 'playable', chen
        return 'trash', chen

    def board_texture(self, board_cards):
        if not board_cards:
            return {
                'paired': False,
                'monotone': False,
                'two_tone': False,
                'wet': False,
                'high': 0,
                'connected': 0
            }
        rs, ss = self.ranks_suits(board_cards)
        cnt_r = Counter(rs)
        cnt_s = Counter(ss)
        uniq = sorted(set(rs))
        paired = max(cnt_r.values()) >= 2
        monotone = max(cnt_s.values()) >= 3
        two_tone = max(cnt_s.values()) >= 2

        connected = 0
        if len(uniq) >= 2:
            mx = 1
            cur = 1
            for i in range(1, len(uniq)):
                if uniq[i] - uniq[i - 1] <= 2:
                    cur += 1
                else:
                    cur = 1
                mx = max(mx, cur)
            connected = mx

        wet = monotone or connected >= 3 or (paired and len(board_cards) >= 4)
        return {
            'paired': paired,
            'monotone': monotone,
            'two_tone': two_tone,
            'wet': wet,
            'high': max(rs),
            'connected': connected
        }

    def straight_draw_info(self, cards):
        rs = {self.rank_map[c[0]] for c in cards}
        if 14 in rs:
            rs.add(1)
        misses = set()
        for st in range(1, 11):
            seq = set(range(st, st + 5))
            have = seq & rs
            if len(have) == 4:
                miss = list(seq - have)[0]
                misses.add(miss)
        if len(misses) >= 2:
            return 'oesd'
        if len(misses) == 1:
            return 'gutshot'
        return 'none'

    def flush_draw_info(self, hole_cards, board_cards):
        cards = hole_cards + board_cards
        suits = Counter([c[1] for c in cards])
        hole_suits = [c[1] for c in hole_cards]
        for s, cnt in suits.items():
            if cnt == 4 and s in hole_suits:
                nut = any(c[1] == s and c[0] == 'A' for c in hole_cards)
                return 'nut_fd' if nut else 'fd'
        return 'none'

    def made_hand_features(self, hole_cards, board_cards):
        all_cards = hole_cards + board_cards
        rs_h, _ = self.ranks_suits(hole_cards)
        rs_b, _ = self.ranks_suits(board_cards)
        v = eval7.evaluate([eval7.Card(c) for c in all_cards])
        ht = eval7.handtype(v)

        top_pair = False
        overpair = False
        pair_plus = ht in {'Pair', 'Two Pair', 'Trips', 'Straight', 'Flush', 'Full House', 'Quads', 'Straight Flush'}

        if board_cards:
            hi_board = max(rs_b)
            if rs_h[0] == rs_h[1] and rs_h[0] > hi_board and ht == 'Pair':
                overpair = True
            if ht == 'Pair' and hi_board in rs_h:
                top_pair = True

        return {
            'handtype': ht,
            'top_pair': top_pair,
            'overpair': overpair,
            'pair_plus': pair_plus,
            'value': v
        }

    def infer_range_string(self, street, continue_cost, pot_size, board_tex, opp_aggr):
        frac = continue_cost / max(1, pot_size)
        if street == 0:
            if continue_cost <= BIG_BLIND:
                return "22+,A2s+,K2s+,Q5s+,J7s+,T7s+,97s+,86s+,75s+,65s,54s,A2o+,K8o+,Q9o+,J9o+,T9o"
            if frac <= 0.45:
                return "22+,A2s+,K5s+,Q8s+,J8s+,T8s+,98s,87s,76s,A7o+,KTo+,QTo+,JTo"
            if frac <= 1.0:
                return "55+,A8s+,KTs+,QTs+,JTs,ATo+,KQo"
            return "99+,AQs+,AKo"

        if continue_cost == 0:
            if board_tex['wet']:
                return "22+,A2s+,K8s+,Q9s+,J9s+,T9s,98s,87s,76s,A9o+,KTo+,QTo+,JTo"
            return "22+,A2s+,K5s+,Q8s+,J8s+,T8s+,98s,87s,A8o+,KTo+,QTo+,JTo"

        if frac <= 0.25:
            self.opp['small_bets_seen'] += 1
            if opp_aggr > 0.56:
                return "22+,A2s+,K2s+,Q6s+,J7s+,T8s+,97s+,87s,76s,A5o+,K9o+,QTo+,JTo"
            return "22+,A2s+,K5s+,Q8s+,J8s+,T8s+,98s,87s,76s,A7o+,KTo+,QTo+,JTo"
        if frac <= 0.75:
            if opp_aggr < 0.32:
                return "55+,A8s+,KTs+,QTs+,JTs,ATo+,KQo"
            return "44+,A5s+,KTs+,QTs+,JTs,ATo+,KQo"
        self.opp['big_bets_seen'] += 1
        if opp_aggr < 0.32:
            return "88+,ATs+,KQs,AQo+"
        return "77+,ATs+,KQs,AQo+"

    def estimate_equity(self, hole_cards, board_cards, villain_range, game_clock):
        hero = tuple(eval7.Card(c) for c in hole_cards)
        board = tuple(eval7.Card(c) for c in board_cards)
        vr = self.get_range(villain_range)

        try:
            if len(board_cards) >= 4:
                return float(eval7.py_hand_vs_range_exact(hero, vr, board))
        except Exception:
            pass

        iters = 3000
        if len(board_cards) == 0:
            iters = 2000
        elif len(board_cards) == 3:
            iters = 3500
        if game_clock < 20:
            iters = min(iters, 1800)
        if game_clock < 8:
            iters = min(iters, 900)

        try:
            return float(eval7.py_hand_vs_range_monte_carlo(hero, vr, board, iters))
        except Exception:
            feats = self.made_hand_features(hole_cards, board_cards)
            ht = feats['handtype']
            base = {
                'High Card': 0.18,
                'Pair': 0.48,
                'Two Pair': 0.72,
                'Trips': 0.82,
                'Straight': 0.87,
                'Flush': 0.91,
                'Full House': 0.96,
                'Quads': 0.985,
                'Straight Flush': 0.995
            }
            eq = base.get(ht, 0.18)
            if self.flush_draw_info(hole_cards, board_cards) != 'none':
                eq += 0.10
            if self.straight_draw_info(hole_cards + board_cards) == 'oesd':
                eq += 0.08
            elif self.straight_draw_info(hole_cards + board_cards) == 'gutshot':
                eq += 0.03
            return min(0.99, eq)

    def opp_foldiness(self):
        f = self.opp['fold_seen']
        s = self.opp['showdowns']
        return (f + 1.0) / (f + s + 2.0)

    def opp_raise_rate(self):
        if self.decision_points < 10:
            return 0.42
        return self.opp_bet_raise_events / max(1, self.decision_points)

    def should_bluff_blocker(self, hole_cards, board_cards):
        if not board_cards:
            return False
        suits = Counter([c[1] for c in board_cards])
        for s, cnt in suits.items():
            if cnt >= 3 and any(c[1] == s and c[0] == 'A' for c in hole_cards):
                return True
        hi_board = max(self.rank_map[c[0]] for c in board_cards)
        return any(self.rank_map[c[0]] == hi_board + 1 for c in hole_cards if hi_board < 14)

    def raise_to(self, round_state, amount):
        mn, mx = round_state.raise_bounds()
        amount = max(mn, min(mx, int(amount)))
        return RaiseAction(amount)

    def bet_from_fraction(self, round_state, active, frac):
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        continue_cost = opp_pip - my_pip
        pot = self.pot_size(round_state)
        target = my_pip + continue_cost + int(frac * (pot + continue_cost))
        return self.raise_to(round_state, target)

    def jam_or_raise(self, round_state, active, frac=0.9):
        _, mx = round_state.raise_bounds()
        my_pip = round_state.pips[active]
        pot = self.pot_size(round_state)
        target = my_pip + int(frac * pot) + BIG_BLIND
        if target >= 0.78 * mx:
            return RaiseAction(mx)
        return self.raise_to(round_state, target)

    def count_raise_events(self, round_state, active):
        ours = 0
        opp = 0
        state = round_state
        while getattr(state, 'previous_state', None) is not None:
            prev = state.previous_state
            actor = prev.button % 2
            now = state.pips[actor]
            old = prev.pips[actor]
            if now > old:
                call_cost = max(0, prev.pips[1 - actor] - prev.pips[actor])
                extra = now - old - call_cost
                if extra > 0:
                    if actor == active:
                        ours += 1
                    else:
                        opp += 1
            state = prev
        return ours, opp

    def board_scare_delta(self, board_cards):
        if len(board_cards) < 4:
            return False
        prev = board_cards[:-1]
        cur = board_cards
        prev_tex = self.board_texture(prev)
        cur_tex = self.board_texture(cur)
        last_rank = self.rank_map[board_cards[-1][0]]
        prev_high = max(self.rank_map[c[0]] for c in prev)
        if last_rank >= 13 and prev_high < 13:
            return True
        if cur_tex['paired'] and not prev_tex['paired']:
            return True
        if cur_tex['wet'] and not prev_tex['wet']:
            return True
        return False

    def boss_mode(self):
        return self.round_id >= 8 and self.opp_foldiness() > 0.54 and self.opp_raise_rate() < 0.50

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = [str(c) for c in round_state.hands[active]]
        board_cards = [str(c) for c in round_state.deck[:street]]

        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        continue_cost = opp_pip - my_pip
        pot = self.pot_size(round_state)
        spr = round_state.stacks[active] / max(1, pot)
        our_raises, opp_raises = self.count_raise_events(round_state, active)
        boss = self.boss_mode()

        self.decision_points += 1
        if continue_cost > 0:
            self.opp_bet_raise_events += 1
        elif street > 0:
            self.hand_opp_checks += 1
            self.opp_check_events += 1

        if getattr(game_state, 'game_clock', 100.0) < 1.0:
            if CheckAction in legal_actions:
                return CheckAction()
            if CallAction in legal_actions and continue_cost <= BIG_BLIND:
                return CallAction()
            return FoldAction() if FoldAction in legal_actions else CheckAction()

        # -------------------- PREFLOP --------------------
        if street == 0:
            cls, chen = self.classify_preflop(my_cards)
            mix = self.mix(my_cards, street)
            opp_aggr = self.opp_raise_rate()
            ace_blocker = any(c[0] == 'A' for c in my_cards)

            if continue_cost == 0:
                if CheckAction in legal_actions and RaiseAction not in legal_actions:
                    return CheckAction()
                if RaiseAction in legal_actions:
                    if cls == 'premium':
                        self.mark_aggr()
                        return self.bet_from_fraction(round_state, active, 1.20)
                    if cls == 'strong':
                        self.mark_aggr()
                        return self.bet_from_fraction(round_state, active, 1.00)
                    if cls == 'playable' and mix < 0.82:
                        self.mark_aggr()
                        return self.bet_from_fraction(round_state, active, 0.85)
                    if cls == 'trash' and opp_aggr < 0.38 and mix < 0.14:
                        self.mark_aggr()
                        return self.bet_from_fraction(round_state, active, 0.70)
                return CheckAction() if CheckAction in legal_actions else FoldAction()

            req = continue_cost / max(1, pot + continue_cost)
            big_raise = continue_cost >= 6 * BIG_BLIND
            tiny_raise = continue_cost <= 2 * BIG_BLIND

            if RaiseAction in legal_actions:
                if cls == 'premium':
                    self.mark_aggr()
                    if chen >= 14 or big_raise:
                        return self.jam_or_raise(round_state, active, 1.1)
                    return self.bet_from_fraction(round_state, active, 1.05)
                if cls == 'strong' and tiny_raise and mix < 0.38:
                    self.mark_aggr()
                    return self.bet_from_fraction(round_state, active, 0.95)
                if cls == 'playable' and tiny_raise and opp_aggr < 0.48 and mix < (0.18 if boss else 0.12):
                    self.mark_aggr()
                    return self.bet_from_fraction(round_state, active, 0.75)
                if tiny_raise and boss and ace_blocker and mix < 0.16:
                    self.mark_aggr()
                    return self.bet_from_fraction(round_state, active, 0.82)

            if CallAction in legal_actions:
                if cls == 'premium':
                    return CallAction()
                if cls == 'strong' and (req <= 0.34 or not big_raise):
                    return CallAction()
                if cls == 'playable' and req <= (0.24 if opp_aggr > 0.55 else 0.22) and not big_raise:
                    return CallAction()
                if cls == 'trash' and tiny_raise and chen >= 6.5 and opp_aggr < 0.46 and mix < 0.08:
                    return CallAction()

            return FoldAction() if FoldAction in legal_actions else CheckAction()

        # -------------------- POSTFLOP --------------------
        board_tex = self.board_texture(board_cards)
        scare = self.board_scare_delta(board_cards)
        opp_aggr = self.opp_raise_rate()
        feats = self.made_hand_features(my_cards, board_cards)
        draw_s = self.straight_draw_info(my_cards + board_cards)
        draw_f = self.flush_draw_info(my_cards, board_cards)
        range_str = self.infer_range_string(street, continue_cost, pot, board_tex, opp_aggr)
        foldy = self.opp_foldiness()
        eq = self.estimate_equity(my_cards, board_cards, range_str, getattr(game_state, 'game_clock', 100.0))
        req = continue_cost / max(1, pot + continue_cost)

        strong_made = feats['handtype'] in {'Two Pair', 'Trips', 'Straight', 'Flush', 'Full House', 'Quads', 'Straight Flush'}
        one_pair_good = feats['top_pair'] or feats['overpair']
        strong_draw = (draw_f != 'none') or (draw_s == 'oesd')
        medium_draw = draw_s == 'gutshot'
        blocker_bluff = self.should_bluff_blocker(my_cards, board_cards)
        mix = self.mix(my_cards, street, 1)

        # checked to us
        if continue_cost == 0:
            if CheckAction in legal_actions and RaiseAction not in legal_actions:
                return CheckAction()

            if RaiseAction in legal_actions:
                if strong_made:
                    self.mark_aggr()
                    if street == 5:
                        return self.bet_from_fraction(round_state, active, 0.95 if board_tex['wet'] else 0.75)
                    return self.bet_from_fraction(round_state, active, 0.75 if board_tex['wet'] else 0.60)

                if one_pair_good:
                    if street == 5:
                        if mix < 0.72:
                            self.mark_aggr()
                            return self.bet_from_fraction(round_state, active, 0.45 if not board_tex['wet'] else 0.58)
                    else:
                        self.mark_aggr()
                        return self.bet_from_fraction(round_state, active, 0.42 if not board_tex['wet'] else 0.55)

                if strong_draw:
                    self.mark_aggr()
                    return self.bet_from_fraction(round_state, active, 0.52 if board_tex['wet'] else 0.38)

                if street == 3:
                    dry_stab = min(0.88, 0.35 + foldy + 0.10 * min(2, self.hand_opp_checks))
                    wet_stab = min(0.52, 0.10 + 0.50 * foldy + 0.06 * self.hand_opp_checks)
                    if not board_tex['wet'] and mix < dry_stab:
                        self.mark_aggr()
                        return self.bet_from_fraction(round_state, active, 0.32)
                    if board_tex['wet'] and blocker_bluff and mix < wet_stab:
                        self.mark_aggr()
                        return self.bet_from_fraction(round_state, active, 0.42)
                elif street == 4:
                    turn_stab = min(0.42, 0.08 + 0.45 * foldy + 0.05 * self.hand_opp_checks)
                    if blocker_bluff and not board_tex['paired'] and mix < turn_stab:
                        self.mark_aggr()
                        return self.bet_from_fraction(round_state, active, 0.48)
                    if boss and scare and blocker_bluff and mix < 0.32:
                        self.mark_aggr()
                        return self.bet_from_fraction(round_state, active, 0.92)
                elif street == 5:
                    if boss and scare and blocker_bluff and not one_pair_good and mix < 0.26:
                        self.mark_aggr()
                        return self.bet_from_fraction(round_state, active, 1.05)
                    if boss and self.hand_opp_checks >= 2 and mix < 0.34:
                        self.mark_aggr()
                        return self.bet_from_fraction(round_state, active, 0.78)

            return CheckAction() if CheckAction in legal_actions else FoldAction()

        # facing a bet
        bet_frac = continue_cost / max(1, pot)
        overbet = bet_frac > 1.0
        tiny_bet = bet_frac <= 0.20
        facing_raise = opp_raises > 0 or my_pip > 0

        if strong_made:
            if RaiseAction in legal_actions:
                self.mark_aggr()
                if street == 5:
                    return self.jam_or_raise(round_state, active, 1.0 if overbet else 0.85)
                if eq > 0.78 or spr < 2.2:
                    return self.jam_or_raise(round_state, active, 0.95)
                return self.bet_from_fraction(round_state, active, 0.80)
            return CallAction() if CallAction in legal_actions else CheckAction()

        if one_pair_good:
            if street == 5 and boss and eq < req + (0.14 if overbet else 0.08):
                return FoldAction() if FoldAction in legal_actions else CheckAction()
            if facing_raise and board_tex['wet'] and eq < req + 0.12:
                return FoldAction() if FoldAction in legal_actions else CheckAction()
            if overbet:
                if eq > req + 0.08 and CallAction in legal_actions:
                    return CallAction()
                return FoldAction() if FoldAction in legal_actions else CheckAction()
            if RaiseAction in legal_actions and tiny_bet and eq > 0.66 and mix < 0.30 and opp_aggr < 0.58:
                self.mark_aggr()
                return self.bet_from_fraction(round_state, active, 0.70)
            if CallAction in legal_actions and (eq > req + 0.03 or tiny_bet):
                return CallAction()

        if strong_draw:
            if street == 5 and boss:
                return FoldAction() if FoldAction in legal_actions else CheckAction()
            if facing_raise and overbet and eq < req + 0.10:
                return FoldAction() if FoldAction in legal_actions else CheckAction()
            if RaiseAction in legal_actions and (eq > req + 0.07 or (blocker_bluff and mix < 0.40 and opp_aggr < 0.60)):
                self.mark_aggr()
                return self.bet_from_fraction(round_state, active, 0.78 if board_tex['wet'] else 0.62)
            if CallAction in legal_actions and eq > req - 0.02:
                return CallAction()

        if medium_draw:
            if tiny_bet and CallAction in legal_actions:
                return CallAction()
            if CallAction in legal_actions and eq > req + 0.01 and not overbet:
                return CallAction()

        if tiny_bet and CallAction in legal_actions:
            high_cards = sorted([self.rank_map[c[0]] for c in my_cards], reverse=True)
            if high_cards[0] >= 13 or high_cards[1] >= 10 or (street in (3, 4) and blocker_bluff):
                return CallAction()

        raw_margin = 0.03 if opp_aggr > 0.58 else 0.05
        if boss and street in (3, 4):
            raw_margin -= 0.015
        if street == 5 and boss:
            raw_margin += 0.03
        if CallAction in legal_actions and eq > req + raw_margin and not overbet:
            return CallAction()

        return FoldAction() if FoldAction in legal_actions else CheckAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
