/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.core.AI
 *  rts.units.UnitTypeTable
 */
package ai.rai;

import ai.core.AI;
import ai.rai.RAISocketAI;
import rts.units.UnitTypeTable;

public class RAISocketAIBestModels
extends RAISocketAI {
    public RAISocketAIBestModels(UnitTypeTable unitTypeTable) {
        super(100, -1, unitTypeTable, 0, 1, true);
    }

    public RAISocketAIBestModels(int n, int n2, UnitTypeTable unitTypeTable) {
        super(n, n2, unitTypeTable, 0, 1, true);
    }

    public RAISocketAIBestModels(int n, int n2, UnitTypeTable unitTypeTable, int n3, int n4) {
        super(n, n2, unitTypeTable, n3, n4, true);
    }

    @Override
    public AI clone() {
        if (DEBUG >= 1) {
            System.out.println("RAISocketAIBestModels: cloning");
        }
        return new RAISocketAIBestModels(this.TIME_BUDGET, this.ITERATIONS_BUDGET, this.utt, this.OVERRIDE_TORCH_THREADS, this.PYTHON_VERBOSE_LEVEL);
    }
}

