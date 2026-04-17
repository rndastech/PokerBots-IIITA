#include <skeleton/actions.h>
#include <skeleton/constants.h>
#include <skeleton/runner.h>
#include <skeleton/states.h>
#include <iostream>
#include <algorithm>
#include <array>
#include <map>
#include <vector>

using namespace pokerbots::skeleton;

int evaluatePreflopStrength(char rank1, char rank2, bool isSuited) {
    auto getPoints = [](char r) {
        if (r == 'A') return 10.0; if (r == 'K') return 8.0;
        if (r == 'Q') return 7.0; if (r == 'J') return 6.0;
        if (r == 'T') return 5.0; return (r - '0') / 2.0; 
    };
    double score = std::max(getPoints(rank1), getPoints(rank2));
    if (rank1 == rank2) score = std::max(score * 2.0, 5.0); 
    else score -= 1.0; 
    if (isSuited) score += 2.0;
    return (int)score;
}

int evaluatePostflopStrength(const std::array<std::string, 2>& hole, const std::array<std::string, 5>& board) {
    std::map<char, int> rankCounts;
    std::map<char, int> suitCounts;

    for (const auto& card : hole) {
        rankCounts[card[0]]++;
        suitCounts[card[1]]++;
    }
    for (const auto& card : board) {
        if (!card.empty()) { 
            rankCounts[card[0]]++;
            suitCounts[card[1]]++;
        }
    }

    int maxSuitCount = 0;
    char flushSuit = ' '; // Track WHICH suit is making the flush
    for (const auto& pair : suitCounts) {
        if (pair.second > maxSuitCount) {
            maxSuitCount = pair.second;
            flushSuit = pair.first;
        }
    }

    bool hitPair = false;
    bool hitTrips = false;

    for (const auto& card : hole) {
        if (rankCounts[card[0]] == 2) hitPair = true;
        if (rankCounts[card[0]] >= 3) hitTrips = true;
    }

    // THE FIX: We only have a "God Tier" flush if WE hold at least one of the flush cards!
    bool weHoldFlushCard = (hole[0][1] == flushSuit || hole[1][1] == flushSuit);
    
    if (maxSuitCount >= 5 && weHoldFlushCard) return 100; // Real Flush
    if (hitTrips) return 80;           // Trips
    if (hitPair) return 50;            // Pair
    if (maxSuitCount == 4 && weHoldFlushCard) return 40;  // Real Flush Draw
    
    return 0; // Missed
}

struct Bot {
  void handleNewRound(GameInfoPtr gameState, RoundStatePtr roundState, int active) {}
  void handleRoundOver(GameInfoPtr gameState, TerminalStatePtr terminalState, int active) {}

  Action getAction(GameInfoPtr gameState, RoundStatePtr roundState, int active) {
    auto legalActions = roundState->legalActions();
    int street = roundState->street;
    auto myCards = roundState->hands[active]; 
    auto boardCards = roundState->deck; 
    int myPip = roundState->pips[active]; 
    int oppPip = roundState->pips[1-active]; 
    int continueCost = oppPip - myPip;

    std::array<int, 2> raiseBounds = {0, 0};
    if (legalActions.find(Action::Type::RAISE) != legalActions.end()) {
      raiseBounds = roundState->raiseBounds(); 
    }

    auto getSafeRaise = [&](int extraChips) {
        int target = raiseBounds[0] + extraChips;
        return std::min(target, raiseBounds[1]); 
    };

    // --- 1. TIERED PREFLOP LOGIC ---
    if (street == 0) { 
        int handScore = evaluatePreflopStrength(myCards[0][0], myCards[1][0], myCards[0][1] == myCards[1][1]);

        if (handScore >= 16) { // AA, KK (The Nuts)
            // No limit. We want all the chips in the middle.
            if (legalActions.find(Action::Type::RAISE) != legalActions.end()) return {Action::Type::RAISE, getSafeRaise(200)}; 
            else if (legalActions.find(Action::Type::CALL) != legalActions.end()) return {Action::Type::CALL};
        } else if (handScore >= 9) { // Good Hands (Aces, Kings)
            // PRICE CAP: Never pay more than 100 to see the flop!
            if (continueCost > 100 && legalActions.find(Action::Type::FOLD) != legalActions.end()) return {Action::Type::FOLD};
            if (legalActions.find(Action::Type::RAISE) != legalActions.end()) return {Action::Type::RAISE, getSafeRaise(50)}; 
            else if (legalActions.find(Action::Type::CALL) != legalActions.end()) return {Action::Type::CALL};
        } else if (handScore >= 5) { // Playable Hands
            // PRICE CAP: Never pay more than 20 chips!
            if (continueCost > 20 && legalActions.find(Action::Type::FOLD) != legalActions.end()) return {Action::Type::FOLD};
            if (legalActions.find(Action::Type::CALL) != legalActions.end()) return {Action::Type::CALL};
        } else { // Garbage
            if (continueCost > 0 && legalActions.find(Action::Type::FOLD) != legalActions.end()) return {Action::Type::FOLD};
        }
    }

   // --- 2. MIXED STRATEGY POSTFLOP LOGIC ---
    if (street > 0) {
        int postflopScore = evaluatePostflopStrength(myCards, boardCards);
        int rng = rand() % 100; // Generate an entropy variable (0 to 99)

        // --- MIXED STRATEGY INJECTIONS ---

        // 1. THE BLUFF (10% frequency)
        // If we missed the board entirely, pretend we have a monster 10% of the time.
        if (postflopScore < 40 && rng < 10) {
            if (legalActions.find(Action::Type::RAISE) != legalActions.end()) {
                return {Action::Type::RAISE, getSafeRaise(150)};
            }
        }

        // 2. THE TRAP (20% frequency)
        // If we have Trips (80+), just call 20% of the time to induce a bluff from the opponent.
        if (postflopScore >= 80 && rng < 20) {
             if (legalActions.find(Action::Type::CALL) != legalActions.end()) {
                 return {Action::Type::CALL};
             }
        }

        // --- CORE LOGIC WITH VARIABLE SIZING ---
        
        if (postflopScore >= 100) { // Flush (The Nuts)
            if (legalActions.find(Action::Type::RAISE) != legalActions.end()) return {Action::Type::RAISE, raiseBounds[1]};
            if (legalActions.find(Action::Type::CALL) != legalActions.end()) return {Action::Type::CALL};
        } 
        else if (postflopScore >= 80) { // Trips
            if (continueCost > 300 && legalActions.find(Action::Type::FOLD) != legalActions.end()) return {Action::Type::FOLD};
            
            // 3. VARIABLE SIZING: Randomize the value bet between 120 and 200 chips.
            int dynamicBet = 120 + (rand() % 81); 
            if (legalActions.find(Action::Type::RAISE) != legalActions.end()) return {Action::Type::RAISE, getSafeRaise(dynamicBet)};
            if (legalActions.find(Action::Type::CALL) != legalActions.end()) return {Action::Type::CALL};
        } 
        else if (postflopScore >= 40) { // Pair
            if (continueCost > 50 && legalActions.find(Action::Type::FOLD) != legalActions.end()) return {Action::Type::FOLD};
            if (legalActions.find(Action::Type::RAISE) != legalActions.end()) return {Action::Type::RAISE, raiseBounds[0]};
            else if (legalActions.find(Action::Type::CALL) != legalActions.end()) return {Action::Type::CALL};
        } 
        else { // Missed Board
            if (continueCost > 0 && legalActions.find(Action::Type::FOLD) != legalActions.end()) return {Action::Type::FOLD}; 
        }
    }

    // --- 3. FAILSAFE ---
    if (legalActions.find(Action::Type::CHECK) != legalActions.end()) return {Action::Type::CHECK};
    if (legalActions.find(Action::Type::CALL) != legalActions.end()) return {Action::Type::CALL};
    return {Action::Type::FOLD}; 
  }
};

int main(int argc, char *argv[]) {
  srand(time(NULL));
  auto [host, port] = parseArgs(argc, argv);
  runBot<Bot>(host, port);
  return 0;
}