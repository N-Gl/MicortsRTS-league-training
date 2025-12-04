/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.reward.RewardFunctionInterface
 *  rts.GameState
 *  rts.TraceEntry
 */
package ai.reward;

import ai.reward.RewardFunctionInterface;
import rts.GameState;
import rts.TraceEntry;

public class RAIWinLossRewardFunction
extends RewardFunctionInterface {
    public void computeReward(int n, int n2, TraceEntry traceEntry, GameState gameState) {
        this.reward = 0.0;
        this.done = false;
        if (gameState.gameover()) {
            this.done = true;
            int n3 = gameState.winner();
            this.reward = n3 == n ? 1.0 : (n3 == n2 ? -1.0 : 0.0);
        }
    }
}

