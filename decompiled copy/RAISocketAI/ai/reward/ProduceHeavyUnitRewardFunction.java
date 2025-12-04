/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.reward.RewardFunctionInterface
 *  rts.GameState
 *  rts.TraceEntry
 *  rts.UnitAction
 *  rts.units.Unit
 *  util.Pair
 */
package ai.reward;

import ai.reward.RewardFunctionInterface;
import rts.GameState;
import rts.TraceEntry;
import rts.UnitAction;
import rts.units.Unit;
import util.Pair;

public class ProduceHeavyUnitRewardFunction
extends RewardFunctionInterface {
    public void computeReward(int n, int n2, TraceEntry traceEntry, GameState gameState) {
        this.reward = 0.0;
        this.done = false;
        for (Pair pair : traceEntry.getActions()) {
            if (((Unit)pair.m_a).getPlayer() != n || ((UnitAction)pair.m_b).getType() != 4 || ((UnitAction)pair.m_b).getUnitType() == null || !((UnitAction)pair.m_b).getUnitType().name.equals("Heavy")) continue;
            this.reward += 1.0;
        }
    }

    public double getReward() {
        return this.reward;
    }

    public boolean isDone() {
        return this.done;
    }
}

