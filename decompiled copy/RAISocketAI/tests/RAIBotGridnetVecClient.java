/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.core.AI
 *  ai.reward.RewardFunctionInterface
 *  rts.units.UnitTypeTable
 */
package tests;

import ai.core.AI;
import ai.rai.RAIBotResponse;
import ai.rai.RAIBotResponses;
import ai.reward.RewardFunctionInterface;
import java.util.Arrays;
import rts.units.UnitTypeTable;
import tests.RAIBotClient;

public class RAIBotGridnetVecClient {
    public RAIBotClient[] botClients;
    public int maxSteps;
    public int[] envSteps;
    public RewardFunctionInterface[] rfs;
    public UnitTypeTable utt;
    boolean partialObs = false;
    public String[] mapPaths;
    byte[][] mask;
    byte[][] observation;
    double[][] reward;
    boolean[][] done;
    byte[][] resources;
    RAIBotResponse[] rs;
    RAIBotResponses responses;
    byte[][] terrain;
    int[][][] action;
    double[] terminalReward1;
    boolean[] terminalDone1;
    int[][] terminalAction1;
    double[] terminalReward2;
    boolean[] terminalDone2;
    int[][] terminalAction2;

    public RAIBotGridnetVecClient(int n, RewardFunctionInterface[] rewardFunctionInterfaceArray, String string, String[] stringArray, AI[] aIArray, UnitTypeTable unitTypeTable, boolean bl) throws Exception {
        int n2;
        this.maxSteps = n;
        this.utt = unitTypeTable;
        this.rfs = rewardFunctionInterfaceArray;
        this.partialObs = bl;
        this.mapPaths = stringArray;
        this.botClients = new RAIBotClient[aIArray.length / 2];
        for (n2 = 0; n2 < this.botClients.length; ++n2) {
            this.botClients[n2] = new RAIBotClient(rewardFunctionInterfaceArray, string, this.mapPaths[2 * n2], aIArray[2 * n2], aIArray[2 * n2 + 1], unitTypeTable, this.partialObs);
        }
        n2 = this.botClients.length * 2;
        this.envSteps = new int[n2];
        this.mask = new byte[n2][];
        this.observation = new byte[n2][];
        this.reward = new double[n2][this.rfs.length];
        this.done = new boolean[n2][this.rfs.length];
        this.resources = new byte[n2][2];
        this.terminalReward1 = new double[this.rfs.length];
        this.terminalDone1 = new boolean[this.rfs.length];
        this.terminalReward2 = new double[this.rfs.length];
        this.terminalDone2 = new boolean[this.rfs.length];
        this.responses = new RAIBotResponses(null, null, null, null, null, null, null);
        this.terrain = new byte[n2][];
        this.action = new int[n2][][];
        this.rs = new RAIBotResponse[n2];
    }

    public RAIBotResponses reset() throws Exception {
        int n;
        for (n = 0; n < this.botClients.length; ++n) {
            this.botClients[n].reset();
            for (int i = 0; i < 2; ++i) {
                this.rs[n * 2 + i] = this.botClients[n].getResponse(i);
            }
        }
        Arrays.fill(this.envSteps, 0);
        for (n = 0; n < this.rs.length; ++n) {
            this.observation[n] = this.rs[n].observation;
            this.mask[n] = this.rs[n].mask;
            this.reward[n] = this.rs[n].reward;
            this.done[n] = this.rs[n].done;
            this.terrain[n] = this.rs[n].terrain;
            this.resources[n] = this.rs[n].resources;
        }
        this.responses.set(this.observation, this.mask, this.reward, this.done, this.terrain, this.resources, null);
        return this.responses;
    }

    public RAIBotResponses gameStep() throws Exception {
        int n;
        for (n = 0; n < this.botClients.length; ++n) {
            int n2;
            this.botClients[n].gameStep();
            for (n2 = 0; n2 < 2; ++n2) {
                this.rs[n * 2 + n2] = this.botClients[n].getResponse(n2);
                int n3 = n * 2 + n2;
                this.envSteps[n3] = this.envSteps[n3] + 1;
            }
            if (!this.rs[n * 2].done[0] && this.envSteps[n * 2] < this.maxSteps) continue;
            for (n2 = 0; n2 < this.terminalReward1.length; ++n2) {
                this.terminalReward1[n2] = this.rs[n * 2].reward[n2];
                this.terminalDone1[n2] = this.rs[n * 2].done[n2];
                this.terminalReward2[n2] = this.rs[n * 2 + 1].reward[n2];
                this.terminalDone2[n2] = this.rs[n * 2 + 1].done[n2];
            }
            this.terminalAction1 = this.rs[n * 2].action;
            this.terminalAction2 = this.rs[n * 2 + 1].action;
            this.botClients[n].reset();
            for (n2 = 0; n2 < this.terminalReward1.length; ++n2) {
                this.rs[n * 2].reward[n2] = this.terminalReward1[n2];
                this.rs[n * 2].done[n2] = this.terminalDone1[n2];
                this.rs[n * 2 + 1].reward[n2] = this.terminalReward2[n2];
                this.rs[n * 2 + 1].done[n2] = this.terminalDone2[n2];
            }
            this.rs[n * 2].done[0] = true;
            this.rs[n * 2 + 1].done[0] = true;
            this.rs[n * 2].action = this.terminalAction1;
            this.rs[n * 2 + 1].action = this.terminalAction2;
            this.envSteps[n * 2] = 0;
            this.envSteps[n * 2 + 1] = 0;
        }
        for (n = 0; n < this.rs.length; ++n) {
            this.observation[n] = this.rs[n].observation;
            this.mask[n] = this.rs[n].mask;
            this.reward[n] = this.rs[n].reward;
            this.done[n] = this.rs[n].done;
            this.resources[n] = this.rs[n].resources;
            this.action[n] = this.rs[n].action;
        }
        this.responses.set(this.observation, this.mask, this.reward, this.done, null, this.resources, this.action);
        return this.responses;
    }

    public void close() throws Exception {
        for (int i = 0; i < this.botClients.length; ++i) {
            this.botClients[i].close();
        }
    }
}

