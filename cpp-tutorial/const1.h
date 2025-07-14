#include <iostream>

using namespace std;

// Version1
// class GamePlayer {
//     static const int numPlayers = 5; // static constant member variable
//     int scores[numPlayers]; // array of scores for each player
// public:
//     GamePlayer() {
//         for (int i = 0; i < numPlayers; ++i) {
//             scores[i] = 0; // initialize scores to 0
//         }
//     }
// };

// Version2
class GamePlayer {
    enum {NumPlayers = 5}; // enum constant
    int scores[NumPlayers]; // array of scores for each player
public:
    GamePlayer() {  
        for (int i = 0; i < NumPlayers; ++i) {
            scores[i] = 0; // initialize scores to 0
        }
    }
};