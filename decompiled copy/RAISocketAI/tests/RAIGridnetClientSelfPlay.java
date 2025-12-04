/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.core.AI
 *  ai.jni.JNIAI
 *  ai.jni.JNIInterface
 *  ai.reward.RewardFunctionInterface
 *  gui.PhysicalGameStateJFrame
 *  gui.PhysicalGameStatePanel
 *  rts.GameState
 *  rts.PartiallyObservableGameState
 *  rts.PhysicalGameState
 *  rts.PlayerAction
 *  rts.TraceEntry
 *  rts.units.UnitTypeTable
 */
package tests;

import ai.core.AI;
import ai.jni.JNIAI;
import ai.jni.JNIInterface;
import ai.rai.GameStateWrapper;
import ai.rai.RAIResponse;
import ai.reward.RewardFunctionInterface;
import gui.PhysicalGameStateJFrame;
import gui.PhysicalGameStatePanel;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.StringWriter;
import java.io.Writer;
import java.nio.file.Paths;
import rts.GameState;
import rts.PartiallyObservableGameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.TraceEntry;
import rts.units.UnitTypeTable;

public class RAIGridnetClientSelfPlay {
    public RewardFunctionInterface[] rfs;
    String micrortsPath;
    public String mapPath;
    public AI ai2;
    UnitTypeTable utt;
    boolean partialObs = false;
    PhysicalGameStateJFrame w;
    public JNIInterface[] ais = new JNIInterface[2];
    public PhysicalGameState pgs;
    public GameState gs;
    public GameState[] playergs = new GameState[2];
    boolean gameover = false;
    boolean layerJSON = true;
    public int renderTheme = PhysicalGameStatePanel.COLORSCHEME_WHITE;
    public int maxAttackRadius;
    public int numPlayers = 2;
    int[][][][] masks = new int[2][][][];
    double[][] rewards = new double[2][];
    boolean[][] dones = new boolean[2][];
    RAIResponse[] response = new RAIResponse[2];
    PlayerAction[] pas = new PlayerAction[2];

    public RAIGridnetClientSelfPlay(RewardFunctionInterface[] rewardFunctionInterfaceArray, String string, String string2, UnitTypeTable unitTypeTable, boolean bl) throws Exception {
        this.micrortsPath = string;
        this.mapPath = string2;
        this.rfs = rewardFunctionInterfaceArray;
        this.utt = unitTypeTable;
        this.partialObs = bl;
        this.maxAttackRadius = this.utt.getMaxAttackRange() * 2 + 1;
        if (this.micrortsPath.length() != 0) {
            this.mapPath = Paths.get(this.micrortsPath, this.mapPath).toString();
        }
        this.pgs = PhysicalGameState.load((String)this.mapPath, (UnitTypeTable)this.utt);
        for (int i = 0; i < this.numPlayers; ++i) {
            this.ais[i] = new JNIAI(100, 0, this.utt);
            this.masks[i] = new int[this.pgs.getHeight()][this.pgs.getWidth()][23 + this.utt.getUnitTypes().size() + this.maxAttackRadius * this.maxAttackRadius];
            this.rewards[i] = new double[this.rfs.length];
            this.dones[i] = new boolean[this.rfs.length];
            this.response[i] = new RAIResponse(null, null, null, null, null, null, null);
        }
    }

    public byte[] render(boolean bl) throws Exception {
        if (this.w == null) {
            this.w = PhysicalGameStatePanel.newVisualizer((GameState)this.gs, (int)640, (int)640, (boolean)this.partialObs, (int)this.renderTheme);
        }
        this.w.setStateCloning(this.gs);
        this.w.repaint();
        if (!bl) {
            return null;
        }
        BufferedImage bufferedImage = new BufferedImage(this.w.getWidth(), this.w.getHeight(), 5);
        this.w.paint(bufferedImage.getGraphics());
        WritableRaster writableRaster = bufferedImage.getRaster();
        DataBufferByte dataBufferByte = (DataBufferByte)writableRaster.getDataBuffer();
        return dataBufferByte.getData();
    }

    public void gameStep(int[][] nArray, int[][] nArray2) throws Exception {
        int n;
        TraceEntry traceEntry = new TraceEntry(this.gs.getPhysicalGameState().clone(), this.gs.getTime());
        for (n = 0; n < this.numPlayers; ++n) {
            this.playergs[n] = this.gs;
            if (this.partialObs) {
                this.playergs[n] = new PartiallyObservableGameState(this.gs, n);
            }
            if (n == 0) {
                this.pas[n] = this.ais[n].getAction(n, this.playergs[0], nArray);
                assert (this.pas[n].getActions().size() == nArray.length);
            } else {
                this.pas[n] = this.ais[n].getAction(n, this.playergs[1], nArray2);
                assert (this.pas[n].getActions().size() == nArray2.length);
            }
            this.gs.issueSafe(this.pas[n]);
            traceEntry.addPlayerAction(this.pas[n].clone());
        }
        this.gameover = this.gs.cycle();
        if (this.gameover) {
            // empty if block
        }
        for (n = 0; n < this.numPlayers; ++n) {
            for (int i = 0; i < this.rfs.length; ++i) {
                this.rfs[i].computeReward(n, 1 - n, traceEntry, this.gs);
                this.rewards[n][i] = this.rfs[i].getReward();
                this.dones[n][i] = this.rfs[i].isDone();
            }
            GameStateWrapper gameStateWrapper = new GameStateWrapper(this.gs);
            this.response[n].set(gameStateWrapper.getArrayObservation(n), gameStateWrapper.getBinaryMask(n), this.rewards[n], this.dones[n], "{}", null, gameStateWrapper.getPlayerResources(n));
        }
    }

    public String sendUTT() throws Exception {
        StringWriter stringWriter = new StringWriter();
        this.utt.toJSON((Writer)stringWriter);
        return ((Object)stringWriter).toString();
    }

    public void reset() throws Exception {
        this.pgs = PhysicalGameState.load((String)this.mapPath, (UnitTypeTable)this.utt);
        this.gs = new GameState(this.pgs, this.utt);
        for (int i = 0; i < this.numPlayers; ++i) {
            this.playergs[i] = this.gs;
            if (this.partialObs) {
                this.playergs[i] = new PartiallyObservableGameState(this.gs, i);
            }
            this.ais[i].reset();
            for (int j = 0; j < this.rewards.length; ++j) {
                this.rewards[i][j] = 0.0;
                this.dones[i][j] = false;
            }
            GameStateWrapper gameStateWrapper = new GameStateWrapper(this.gs);
            this.response[i].set(gameStateWrapper.getArrayObservation(i), gameStateWrapper.getBinaryMask(i), this.rewards[i], this.dones[i], "{}", gameStateWrapper.getTerrain(), gameStateWrapper.getPlayerResources(i));
        }
    }

    public RAIResponse getResponse(int n) {
        return this.response[n];
    }

    public void close() throws Exception {
        if (this.w != null) {
            System.out.println(this.getClass().getSimpleName() + ": Not disposing frame. Resource Leak!");
        }
    }
}

