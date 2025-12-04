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
 *  rts.units.Unit
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
import rts.units.Unit;
import rts.units.UnitTypeTable;

public class RAIGridnetClient {
    public RewardFunctionInterface[] rfs;
    String micrortsPath;
    public String mapPath;
    public AI ai2;
    UnitTypeTable utt;
    public boolean partialObs = false;
    public PhysicalGameState pgs;
    public GameState gs;
    public GameState player1gs;
    public GameState player2gs;
    boolean gameover = false;
    boolean layerJSON = true;
    public int renderTheme = PhysicalGameStatePanel.COLORSCHEME_WHITE;
    public int maxAttackRadius;
    PhysicalGameStateJFrame w;
    public JNIInterface ai1;
    int[][][] masks;
    double[] rewards;
    boolean[] dones;
    RAIResponse response;
    PlayerAction pa1;
    PlayerAction pa2;

    public RAIGridnetClient(RewardFunctionInterface[] rewardFunctionInterfaceArray, String string, String string2, AI aI, UnitTypeTable unitTypeTable, boolean bl) throws Exception {
        this.micrortsPath = string;
        this.mapPath = string2;
        this.rfs = rewardFunctionInterfaceArray;
        this.utt = unitTypeTable;
        this.partialObs = bl;
        this.maxAttackRadius = this.utt.getMaxAttackRange() * 2 + 1;
        this.ai1 = new JNIAI(100, 0, this.utt);
        this.ai2 = aI;
        if (this.ai2 == null) {
            throw new Exception("no ai2 was chosen");
        }
        if (this.micrortsPath.length() != 0) {
            this.mapPath = Paths.get(this.micrortsPath, this.mapPath).toString();
        }
        this.pgs = PhysicalGameState.load((String)this.mapPath, (UnitTypeTable)this.utt);
        this.masks = new int[this.pgs.getHeight()][this.pgs.getWidth()][23 + this.utt.getUnitTypes().size() + this.maxAttackRadius * this.maxAttackRadius];
        this.rewards = new double[this.rfs.length];
        this.dones = new boolean[this.rfs.length];
        this.response = new RAIResponse(null, null, null, null, null, null, null);
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

    public RAIResponse gameStep(int[][] nArray, int n) throws Exception {
        if (this.partialObs) {
            this.player1gs = new PartiallyObservableGameState(this.gs, n);
            this.player2gs = new PartiallyObservableGameState(this.gs, 1 - n);
        } else {
            this.player1gs = this.gs;
            this.player2gs = this.gs;
        }
        this.pa1 = this.ai1.getAction(n, this.player1gs, nArray);
        assert (this.pa1.getActions().size() == nArray.length);
        try {
            this.pa2 = this.ai2.getAction(1 - n, this.player2gs);
        }
        catch (Exception exception) {
            exception.printStackTrace();
            this.pa2 = new PlayerAction();
            this.pa2.fillWithNones(this.player2gs, 1 - n, 1);
        }
        TraceEntry traceEntry = new TraceEntry(this.gs.getPhysicalGameState().clone(), this.gs.getTime());
        this.gs.issueSafe(this.pa1);
        traceEntry.addPlayerAction(this.pa1.clone());
        boolean bl = false;
        try {
            this.gs.issueSafe(this.pa2);
            traceEntry.addPlayerAction(this.pa2.clone());
        }
        catch (Throwable throwable) {
            Unit[] unitArray;
            System.err.println("Player 2 Error: " + throwable.getMessage());
            throwable.printStackTrace();
            for (Unit unit2 : unitArray = (Unit[])this.gs.getUnits().stream().filter(unit -> unit.getPlayer() == 1 - n).toArray(Unit[]::new)) {
                this.gs.removeUnit(unit2);
            }
            bl = true;
        }
        if (!bl) {
            this.gameover = this.gs.cycle();
        }
        if (this.gameover || bl) {
            this.ai2.gameOver(this.gs.winner());
            this.gameover = true;
        }
        for (int i = 0; i < this.rewards.length; ++i) {
            this.rfs[i].computeReward(n, 1 - n, traceEntry, this.gs);
            this.dones[i] = this.rfs[i].isDone();
            this.rewards[i] = this.rfs[i].getReward();
        }
        GameStateWrapper gameStateWrapper = new GameStateWrapper(this.gs);
        this.response.set(gameStateWrapper.getArrayObservation(n), gameStateWrapper.getBinaryMask(n), this.rewards, this.dones, this.ai1.computeInfo(n, this.player2gs), null, gameStateWrapper.getPlayerResources(n));
        return this.response;
    }

    public String sendUTT() throws Exception {
        StringWriter stringWriter = new StringWriter();
        this.utt.toJSON((Writer)stringWriter);
        return ((Object)stringWriter).toString();
    }

    public RAIResponse reset(int n) throws Exception {
        this.ai1.reset();
        this.ai2 = this.ai2.clone();
        this.ai2.reset();
        this.pgs = PhysicalGameState.load((String)this.mapPath, (UnitTypeTable)this.utt);
        this.gs = new GameState(this.pgs, this.utt);
        this.player1gs = this.partialObs ? new PartiallyObservableGameState(this.gs, n) : this.gs;
        for (int i = 0; i < this.rewards.length; ++i) {
            this.rewards[i] = 0.0;
            this.dones[i] = false;
        }
        GameStateWrapper gameStateWrapper = new GameStateWrapper(this.gs);
        this.response.set(gameStateWrapper.getArrayObservation(n), gameStateWrapper.getBinaryMask(n), this.rewards, this.dones, "{}", gameStateWrapper.getTerrain(), gameStateWrapper.getPlayerResources(n));
        return this.response;
    }

    public void close() throws Exception {
        if (this.w != null) {
            System.out.println(this.getClass().getSimpleName() + ": Not disposing frame. Resource Leak!");
        }
    }
}

