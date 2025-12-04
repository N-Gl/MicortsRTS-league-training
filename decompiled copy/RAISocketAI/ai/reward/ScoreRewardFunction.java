/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.reward.RewardFunctionInterface
 *  rts.GameState
 *  rts.TraceEntry
 *  rts.units.Unit
 */
package ai.reward;

import ai.reward.RewardFunctionInterface;
import rts.GameState;
import rts.TraceEntry;
import rts.units.Unit;

public class ScoreRewardFunction
extends RewardFunctionInterface {
    public void computeReward(int n, int n2, TraceEntry traceEntry, GameState gameState) {
        this.reward = 0.0;
        this.done = gameState.gameover();
        double d = 0.0;
        double d2 = 0.0;
        for (Unit unit : gameState.getUnits()) {
            double d3 = (double)unit.getCost() * (1.0 + (double)unit.getHitPoints() / (double)unit.getMaxHitPoints());
            if (unit.getPlayer() == n) {
                d += d3;
                continue;
            }
            if (unit.getPlayer() != n2) continue;
            d2 += d3;
        }
        this.reward = (d - d2) / (d + d2 + 1.0);
    }
}

