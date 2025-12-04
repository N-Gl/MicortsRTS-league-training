/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.reward.RewardFunctionInterface
 *  rts.GameState
 *  rts.TraceEntry
 *  rts.UnitAction
 *  rts.units.Unit
 *  rts.units.UnitType
 *  util.Pair
 */
package ai.reward;

import ai.reward.RewardFunctionInterface;
import rts.GameState;
import rts.TraceEntry;
import rts.UnitAction;
import rts.units.Unit;
import rts.units.UnitType;
import util.Pair;

public class ProduceCombatUnitCostRewardFunction
extends RewardFunctionInterface {
    public void computeReward(int n, int n2, TraceEntry traceEntry, GameState gameState) {
        this.reward = 0.0;
        this.done = false;
        for (Pair pair : traceEntry.getActions()) {
            if (((Unit)pair.m_a).getPlayer() != n || ((UnitAction)pair.m_b).getType() != 4 || ((UnitAction)pair.m_b).getUnitType() == null) continue;
            UnitType unitType = ((UnitAction)pair.m_b).getUnitType();
            if (!unitType.name.equals("Light") && !unitType.name.equals("Heavy") && !unitType.name.equals("Ranged")) continue;
            this.reward += (double)unitType.cost;
        }
    }

    public double getReward() {
        return this.reward;
    }

    public boolean isDone() {
        return this.done;
    }
}

