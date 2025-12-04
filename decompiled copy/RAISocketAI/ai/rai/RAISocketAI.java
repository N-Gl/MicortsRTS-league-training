/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.core.AI
 *  ai.core.AIWithComputationBudget
 *  ai.core.ParameterSpecification
 *  rts.GameState
 *  rts.PhysicalGameState
 *  rts.PlayerAction
 *  rts.units.UnitTypeTable
 */
package ai.rai;

import ai.core.AI;
import ai.core.AIWithComputationBudget;
import ai.core.ParameterSpecification;
import ai.rai.GameStateWrapper;
import ai.rai.PreGameAnalysisResponse;
import ai.rai.RAISocketMessageType;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringWriter;
import java.io.Writer;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import rts.GameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.units.UnitTypeTable;

public class RAISocketAI
extends AIWithComputationBudget {
    public static int DEBUG = 0;
    public int PYTHON_VERBOSE_LEVEL = 1;
    public int OVERRIDE_TORCH_THREADS = 0;
    public boolean USE_BEST_MODELS = false;
    public String MODEL_SET = "RAISocketAI";
    UnitTypeTable utt;
    int maxAttackDiameter;
    int expectedStepMilliseconds = 0;
    static Process pythonProcess;
    static BufferedReader inPipe;
    static DataOutputStream outPipe;
    static ThreadPoolExecutor executor;
    static Future<String> pendingRequestHandler;
    boolean sentInitialMapInformation;

    public RAISocketAI(UnitTypeTable unitTypeTable) {
        this(100, -1, unitTypeTable, 0, 1, false, "RAISocketAI");
    }

    public RAISocketAI(int n, int n2, UnitTypeTable unitTypeTable) {
        this(n, n2, unitTypeTable, 0, 1, false, "RAISocketAI");
    }

    public RAISocketAI(int n, int n2, UnitTypeTable unitTypeTable, int n3, int n4, boolean bl) {
        this(n, n2, unitTypeTable, n3, n4, bl, "RAISocketAI");
    }

    public RAISocketAI(int n, int n2, UnitTypeTable unitTypeTable, int n3, int n4, boolean bl, String string) {
        super(n, n2);
        this.utt = unitTypeTable;
        this.OVERRIDE_TORCH_THREADS = n3;
        this.PYTHON_VERBOSE_LEVEL = n4;
        this.USE_BEST_MODELS = bl;
        this.MODEL_SET = string;
        this.maxAttackDiameter = this.utt.getMaxAttackRange() * 2 + 1;
        try {
            this.connectChildProcess();
        }
        catch (Exception exception) {
            exception.printStackTrace();
        }
    }

    public void connectChildProcess() throws Exception {
        if (pythonProcess != null) {
            return;
        }
        ArrayList<String> arrayList = new ArrayList<String>(Arrays.asList("rai_microrts", "--time_budget_ms", String.valueOf(this.TIME_BUDGET), "--override_torch_threads", String.valueOf(this.OVERRIDE_TORCH_THREADS), "--model_set", this.MODEL_SET));
        if (this.PYTHON_VERBOSE_LEVEL > 0) {
            arrayList.add("-" + "v".repeat(this.PYTHON_VERBOSE_LEVEL));
        }
        if (this.USE_BEST_MODELS) {
            arrayList.add("--use_best_models");
        }
        ProcessBuilder processBuilder = new ProcessBuilder(new String[0]);
        processBuilder.command(arrayList);
        pythonProcess = processBuilder.start();
        Runtime.getRuntime().addShutdownHook(new Thread(() -> pythonProcess.destroy()));
        inPipe = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream()));
        outPipe = new DataOutputStream(pythonProcess.getOutputStream());
        executor = new ThreadPoolExecutor(0, 2, 5000L, TimeUnit.MILLISECONDS, new LinkedBlockingDeque<Runnable>());
        this.reset();
    }

    private void pauseChildProcess() {
        block4: {
            if (pythonProcess == null) {
                return;
            }
            if (DEBUG >= 1) {
                System.out.println("RAISocketAI: Pausing Python process");
            }
            try {
                new ProcessBuilder("kill", "-STOP", String.valueOf(pythonProcess.pid())).start();
            }
            catch (IOException iOException) {
                if (DEBUG < 1) break block4;
                iOException.printStackTrace();
            }
        }
    }

    private void resumeChildProcess() {
        block4: {
            if (pythonProcess == null) {
                return;
            }
            if (DEBUG >= 1) {
                System.out.println("RAISocketAI: Resuming Python process");
            }
            try {
                new ProcessBuilder("kill", "-CONT", String.valueOf(pythonProcess.pid())).start();
            }
            catch (IOException iOException) {
                if (DEBUG < 1) break block4;
                iOException.printStackTrace();
            }
        }
    }

    public void send(RAISocketMessageType rAISocketMessageType, byte[][] byArray2) throws Exception {
        int n = (2 + byArray2.length) * 4 + Arrays.stream(byArray2).mapToInt(byArray -> ((byte[])byArray).length).sum();
        ByteBuffer byteBuffer = ByteBuffer.allocate(n);
        byteBuffer.putInt(rAISocketMessageType.ordinal());
        byteBuffer.putInt(byArray2.length);
        for (byte[] byArray3 : byArray2) {
            byteBuffer.putInt(byArray3.length);
        }
        for (byte[] byArray3 : byArray2) {
            byteBuffer.put(byArray3);
        }
        this.send(byteBuffer.array());
    }

    public void send(byte[] byArray) throws Exception {
        outPipe.writeInt(byArray.length);
        outPipe.write(byArray);
        outPipe.flush();
    }

    public String request(RAISocketMessageType rAISocketMessageType, byte[][] byArray) throws Exception {
        return this.request(rAISocketMessageType, byArray, null);
    }

    public String request(RAISocketMessageType rAISocketMessageType, byte[][] byArray, Long l) throws Exception {
        long l2 = System.currentTimeMillis();
        if (pendingRequestHandler != null) {
            this.resumeChildProcess();
            try {
                if (l != null) {
                    pendingRequestHandler.get(l, TimeUnit.MILLISECONDS);
                } else {
                    pendingRequestHandler.get();
                }
                pendingRequestHandler = null;
            }
            catch (TimeoutException timeoutException) {
                if (DEBUG >= 1) {
                    System.out.println("RAISocketAI: Prior request exceeded new timeout!");
                }
                this.pauseChildProcess();
                return null;
            }
            catch (InterruptedException | ExecutionException exception) {
                if (DEBUG >= 1) {
                    System.out.println("RAISocketAI: Prior request errored:");
                    exception.printStackTrace();
                }
                pendingRequestHandler = null;
            }
            if (l != null) {
                l = l - (System.currentTimeMillis() - l2);
                if (DEBUG >= 1) {
                    System.out.println("RAISocketAI: Time remaining " + l + " ms");
                }
                if (l <= 0L) {
                    return null;
                }
                if (this.expectedStepMilliseconds > 0 && (double)l.longValue() < (double)this.expectedStepMilliseconds * 1.1) {
                    System.out.println("RAISocketAI: Time remaining " + l + " ms is not enough larger than expected time " + this.expectedStepMilliseconds + " ms. Skipping turn.");
                    return null;
                }
            }
        }
        this.send(rAISocketMessageType, byArray);
        if (DEBUG >= 2) {
            System.out.println("RAISocketAI: sent " + rAISocketMessageType.name());
        }
        Callable<String> callable = () -> {
            String string = inPipe.readLine();
            if (DEBUG >= 2) {
                System.out.println("RAISocketAI: received response to " + rAISocketMessageType.name());
            }
            return string;
        };
        if (l == null) {
            return callable.call();
        }
        if (DEBUG >= 1 && executor.getPoolSize() == 0) {
            System.out.println("No active threads in executor. Creating thread.");
        }
        pendingRequestHandler = executor.submit(callable);
        try {
            String string = pendingRequestHandler.get(l, TimeUnit.MILLISECONDS);
            pendingRequestHandler = null;
            return string;
        }
        catch (TimeoutException timeoutException) {
            if (DEBUG >= 1) {
                System.out.println("RAISocketAI: Request timed out");
            }
            this.pauseChildProcess();
            return null;
        }
        catch (InterruptedException | ExecutionException exception) {
            if (DEBUG >= 1) {
                System.out.println("RAISocketAI: Exception thrown");
                exception.printStackTrace();
            }
            pendingRequestHandler = null;
            return null;
        }
    }

    public void reset() {
        try {
            this.sentInitialMapInformation = false;
            StringWriter stringWriter = new StringWriter();
            this.utt.toJSON((Writer)stringWriter);
            this.request(RAISocketMessageType.UTT, new byte[][]{stringWriter.toString().getBytes(StandardCharsets.UTF_8)});
        }
        catch (Exception exception) {
            exception.printStackTrace();
        }
    }

    public PlayerAction getAction(int n, GameState gameState) throws Exception {
        PlayerAction playerAction;
        String string;
        long l = System.currentTimeMillis();
        GameStateWrapper gameStateWrapper = new GameStateWrapper(gameState, DEBUG);
        Gson gson = new Gson();
        ArrayList arrayList = new ArrayList(Arrays.asList(gameStateWrapper.getArrayObservation(n), gameStateWrapper.getBinaryMask(n), gameStateWrapper.getPlayerResources(n)));
        if (!this.sentInitialMapInformation || DEBUG >= 1) {
            this.sentInitialMapInformation = true;
            PhysicalGameState physicalGameState = gameState.getPhysicalGameState();
            arrayList.add(new byte[]{(byte)physicalGameState.getHeight(), (byte)physicalGameState.getWidth()});
            arrayList.add(gameStateWrapper.getTerrain());
            if (DEBUG >= 1) {
                arrayList.add(gson.toJson(gameStateWrapper.getVectorObservation(n)).getBytes(StandardCharsets.UTF_8));
                arrayList.add(gson.toJson(gameStateWrapper.getMasks(n)).getBytes(StandardCharsets.UTF_8));
            }
        }
        long l2 = (long)this.TIME_BUDGET - (System.currentTimeMillis() - l);
        if (DEBUG >= 2) {
            System.out.println("RAISocketAI: Remaining time budget: " + l2);
        }
        if ((string = this.request(RAISocketMessageType.GET_ACTION, (byte[][])arrayList.toArray((T[])new byte[0][]), l2)) != null) {
            Type type = new TypeToken<int[][]>(){}.getType();
            int[][] nArray = (int[][])gson.fromJson(string, type);
            playerAction = PlayerAction.fromVectorAction((int[][])nArray, (GameState)gameState, (UnitTypeTable)this.utt, (int)n, (int)this.maxAttackDiameter);
        } else {
            System.out.println("RAISocketAI: Empty getAction response (likely timeout). Returning empty action");
            playerAction = new PlayerAction();
        }
        playerAction.fillWithNones(gameState, n, 1);
        return playerAction;
    }

    public void preGameAnalysis(GameState gameState, long l, String string) throws Exception {
        Object object;
        GameStateWrapper gameStateWrapper = new GameStateWrapper(gameState, DEBUG);
        PhysicalGameState physicalGameState = gameState.getPhysicalGameState();
        ArrayList arrayList = new ArrayList(Arrays.asList(gameStateWrapper.getArrayObservation(0), gameStateWrapper.getBinaryMask(0), gameStateWrapper.getPlayerResources(0), {(byte)physicalGameState.getHeight(), (byte)physicalGameState.getWidth()}, gameStateWrapper.getTerrain(), ByteBuffer.allocate(8).putLong(l).array(), string.getBytes(StandardCharsets.UTF_8)));
        if (DEBUG >= 1) {
            object = new Gson();
            arrayList.add(((Gson)object).toJson(gameStateWrapper.getVectorObservation(0)).getBytes(StandardCharsets.UTF_8));
            arrayList.add(((Gson)object).toJson(gameStateWrapper.getMasks(0)).getBytes(StandardCharsets.UTF_8));
        }
        if ((object = this.request(RAISocketMessageType.PRE_GAME_ANALYSIS, (byte[][])arrayList.toArray((T[])new byte[0][]))) != null) {
            Gson gson = new Gson();
            PreGameAnalysisResponse preGameAnalysisResponse = gson.fromJson((String)object, PreGameAnalysisResponse.class);
            this.expectedStepMilliseconds = preGameAnalysisResponse.e;
            if (DEBUG >= 1) {
                System.out.println("RAISocketAI: Expected step time: " + this.expectedStepMilliseconds + " ms");
            }
        } else {
            this.expectedStepMilliseconds = 0;
        }
        this.sentInitialMapInformation = true;
    }

    public void gameOver(int n) throws Exception {
        this.request(RAISocketMessageType.GAME_OVER, new byte[][]{{(byte)n}});
    }

    public AI clone() {
        if (DEBUG >= 1) {
            System.out.println("RAISocketAI: cloning");
        }
        return new RAISocketAI(this.TIME_BUDGET, this.ITERATIONS_BUDGET, this.utt, this.OVERRIDE_TORCH_THREADS, this.PYTHON_VERBOSE_LEVEL, this.USE_BEST_MODELS, this.MODEL_SET);
    }

    public List<ParameterSpecification> getParameters() {
        ArrayList<ParameterSpecification> arrayList = new ArrayList<ParameterSpecification>();
        return arrayList;
    }
}

