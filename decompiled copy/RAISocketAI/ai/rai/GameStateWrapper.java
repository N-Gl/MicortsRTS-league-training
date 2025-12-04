/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  rts.GameState
 *  rts.PhysicalGameState
 *  rts.Player
 *  rts.PlayerAction
 *  rts.ResourceUsage
 *  rts.UnitAction
 *  rts.UnitActionAssignment
 *  rts.units.Unit
 *  rts.units.UnitType
 *  rts.units.UnitTypeTable
 *  util.Pair
 */
package ai.rai;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import rts.GameState;
import rts.PhysicalGameState;
import rts.Player;
import rts.PlayerAction;
import rts.ResourceUsage;
import rts.UnitAction;
import rts.UnitActionAssignment;
import rts.units.Unit;
import rts.units.UnitType;
import rts.units.UnitTypeTable;
import util.Pair;

public class GameStateWrapper {
    GameState gs;
    int debugLevel;
    boolean allowMoveAwayAttacks;
    ResourceUsage base_ru;
    int[][][][] vectorObservation;
    public static final int numVectorObservationFeatureMaps = 13;
    public static final int numArrayObservationFeatureMaps = 14;
    int[][][][] masks;

    public GameStateWrapper(GameState gameState) {
        this(gameState, 0);
    }

    public GameStateWrapper(GameState gameState, int n) {
        this(gameState, n, false);
    }

    public GameStateWrapper(GameState gameState, int n, boolean bl) {
        this.gs = gameState;
        this.debugLevel = n;
        this.allowMoveAwayAttacks = bl;
        this.base_ru = new ResourceUsage();
        PhysicalGameState physicalGameState = this.gs.getPhysicalGameState();
        for (Unit unit : physicalGameState.getUnits()) {
            UnitActionAssignment unitActionAssignment = this.gs.getActionAssignment(unit);
            if (unitActionAssignment == null) continue;
            ResourceUsage resourceUsage = unitActionAssignment.action.resourceUsage(unit, physicalGameState);
            this.base_ru.merge(resourceUsage);
        }
    }

    public int[][][] getVectorObservation(int n) {
        int n2;
        int n3;
        PhysicalGameState physicalGameState = this.gs.getPhysicalGameState();
        int n4 = physicalGameState.getHeight();
        int n5 = physicalGameState.getWidth();
        if (this.vectorObservation == null) {
            this.vectorObservation = new int[2][13][n4][n5];
        }
        for (int i = 0; i < 13; ++i) {
            n3 = i == 2 || i == 3 ? -1 : (i == 11 ? -128 : 0);
            for (int j = 0; j < this.vectorObservation[n][0].length; ++j) {
                Arrays.fill(this.vectorObservation[n][i][j], n3);
            }
        }
        List list = physicalGameState.getUnits();
        for (n3 = 0; n3 < list.size(); ++n3) {
            Unit unit = (Unit)list.get(n3);
            UnitActionAssignment unitActionAssignment = this.gs.getActionAssignment(unit);
            this.vectorObservation[n][0][unit.getY()][unit.getX()] = unit.getHitPoints();
            this.vectorObservation[n][1][unit.getY()][unit.getX()] = unit.getResources();
            n2 = unit.getPlayer();
            if (n2 != -1) {
                this.vectorObservation[n][2][unit.getY()][unit.getX()] = (n2 + n) % 2;
            }
            this.vectorObservation[n][3][unit.getY()][unit.getX()] = unit.getType().ID;
            if (unitActionAssignment != null) {
                int n6;
                int n7;
                this.vectorObservation[n][4][unit.getY()][unit.getX()] = n7 = unitActionAssignment.action.getType();
                switch (n7) {
                    case 0: {
                        break;
                    }
                    case 1: {
                        this.vectorObservation[n][5][unit.getY()][unit.getX()] = unitActionAssignment.action.getDirection() + 1;
                        break;
                    }
                    case 2: {
                        this.vectorObservation[n][6][unit.getY()][unit.getX()] = unitActionAssignment.action.getDirection() + 1;
                        break;
                    }
                    case 3: {
                        this.vectorObservation[n][7][unit.getY()][unit.getX()] = unitActionAssignment.action.getDirection() + 1;
                        break;
                    }
                    case 4: {
                        this.vectorObservation[n][8][unit.getY()][unit.getX()] = unitActionAssignment.action.getDirection() + 1;
                        this.vectorObservation[n][9][unit.getY()][unit.getX()] = unitActionAssignment.action.getUnitType().ID + 1;
                    }
                    case 5: {
                        n6 = unitActionAssignment.action.getLocationX() - unit.getX();
                        int n8 = unitActionAssignment.action.getLocationY() - unit.getY();
                        int n9 = 4;
                        if (n6 == 0) {
                            if (n8 == -1) {
                                n9 = 0;
                            } else if (n8 == 1) {
                                n9 = 2;
                            }
                        } else if (n8 == 0) {
                            if (n6 == -1) {
                                n9 = 3;
                            } else if (n6 == 1) {
                                n9 = 1;
                            }
                        }
                        this.vectorObservation[n][10][unit.getY()][unit.getX()] = n9 + 1;
                    }
                }
                n6 = unitActionAssignment.time + unitActionAssignment.action.ETA(unit) - this.gs.getTime();
                this.vectorObservation[n][11][unit.getY()][unit.getX()] = GameStateWrapper.byteClampValue(n6);
                continue;
            }
            this.vectorObservation[n][4][unit.getY()][unit.getX()] = 0;
        }
        for (n3 = 0; n3 < this.vectorObservation[n][2].length; ++n3) {
            int n10 = 0;
            while (n10 < this.vectorObservation[n][2][n3].length) {
                int[] nArray = this.vectorObservation[n][3][n3];
                int n11 = n10;
                nArray[n11] = nArray[n11] + 1;
                int[] nArray2 = this.vectorObservation[n][2][n3];
                int n12 = n10++;
                nArray2[n12] = nArray2[n12] + 1;
            }
        }
        for (n3 = 0; n3 < n4; ++n3) {
            for (int i = 0; i < n5; ++i) {
                this.vectorObservation[n][12][n3][i] = 1 - physicalGameState.getTerrain(i, n3);
            }
        }
        for (Integer n13 : this.base_ru.getPositionsUsed()) {
            int n14 = n13 / physicalGameState.getWidth();
            n2 = n13 % physicalGameState.getWidth();
            this.vectorObservation[n][3][n14][n2] = 8;
        }
        return this.vectorObservation[n];
    }

    public byte[] getTerrain() {
        PhysicalGameState physicalGameState = this.gs.getPhysicalGameState();
        int n = physicalGameState.getHeight();
        int n2 = physicalGameState.getWidth();
        byte[] byArray = new byte[n * n2];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n2; ++j) {
                byArray[i * n2 + j] = (byte)(1 - physicalGameState.getTerrain(j, i));
            }
        }
        return byArray;
    }

    public byte[] getArrayObservation(int n) {
        int n2;
        int n3;
        Object object;
        int n4;
        PhysicalGameState physicalGameState = this.gs.getPhysicalGameState();
        List list = physicalGameState.getUnits();
        List list2 = this.base_ru.getPositionsUsed();
        byte[] byArray = new byte[(list.size() + list2.size()) * 14];
        for (n4 = 0; n4 < list.size(); ++n4) {
            object = (Unit)list.get(n4);
            n3 = n4 * 14;
            byArray[n3 + 0] = (byte)object.getY();
            byArray[n3 + 1] = (byte)object.getX();
            byArray[n3 + 2] = (byte)object.getHitPoints();
            byArray[n3 + 3] = (byte)object.getResources();
            n2 = object.getPlayer();
            if (n2 != -1) {
                byArray[n3 + 4] = (byte)((n2 + n) % 2 + 1);
            }
            byArray[n3 + 5] = (byte)(object.getType().ID + 1);
            UnitActionAssignment unitActionAssignment = this.gs.getActionAssignment((Unit)object);
            if (unitActionAssignment != null) {
                int n5;
                byte by;
                byArray[n3 + 6] = by = (byte)unitActionAssignment.action.getType();
                switch (by) {
                    case 0: {
                        break;
                    }
                    case 1: {
                        byArray[n3 + 7] = (byte)(unitActionAssignment.action.getDirection() + 1);
                        break;
                    }
                    case 2: {
                        byArray[n3 + 8] = (byte)(unitActionAssignment.action.getDirection() + 1);
                        break;
                    }
                    case 3: {
                        byArray[n3 + 9] = (byte)(unitActionAssignment.action.getDirection() + 1);
                        break;
                    }
                    case 4: {
                        byArray[n3 + 10] = (byte)(unitActionAssignment.action.getDirection() + 1);
                        byArray[n3 + 11] = (byte)(unitActionAssignment.action.getUnitType().ID + 1);
                    }
                    case 5: {
                        n5 = unitActionAssignment.action.getLocationX() - object.getX();
                        int n6 = unitActionAssignment.action.getLocationY() - object.getY();
                        int n7 = 4;
                        if (n5 == 0) {
                            if (n6 == -1) {
                                n7 = 0;
                            } else if (n6 == 1) {
                                n7 = 2;
                            }
                        } else if (n6 == 0) {
                            if (n5 == -1) {
                                n7 = 3;
                            } else if (n5 == 1) {
                                n7 = 1;
                            }
                        }
                        byArray[n3 + 12] = (byte)(n7 + 1);
                    }
                }
                n5 = unitActionAssignment.time + unitActionAssignment.action.ETA((Unit)object) - this.gs.getTime();
                byArray[n3 + 13] = GameStateWrapper.byteClampValue(n5);
                continue;
            }
            byArray[n3 + 6] = 0;
            byArray[n3 + 13] = GameStateWrapper.byteClampValue(0);
        }
        for (n4 = 0; n4 < list2.size(); ++n4) {
            object = (Integer)list2.get(n4);
            n3 = (n4 + list.size()) * 14;
            n2 = (Integer)object / physicalGameState.getWidth();
            int n8 = (Integer)object % physicalGameState.getWidth();
            byArray[n3 + 0] = (byte)n2;
            byArray[n3 + 1] = (byte)n8;
            byArray[n3 + 5] = 8;
            byArray[n3 + 13] = GameStateWrapper.byteClampValue(0);
        }
        return byArray;
    }

    public int[][][] getMasks(int n) {
        UnitTypeTable unitTypeTable = this.gs.getUnitTypeTable();
        PhysicalGameState physicalGameState = this.gs.getPhysicalGameState();
        int n2 = unitTypeTable.getMaxAttackRange() * 2 + 1;
        if (this.masks == null) {
            this.masks = new int[2][physicalGameState.getHeight()][physicalGameState.getWidth()][23 + unitTypeTable.getUnitTypes().size() + n2 * n2];
        }
        Arrays.stream(this.masks[n]).forEach(nArray2 -> Arrays.stream(nArray2).forEach(nArray -> Arrays.fill(nArray, 0)));
        for (Unit unit : physicalGameState.getUnits()) {
            UnitActionAssignment unitActionAssignment = this.gs.getActionAssignment(unit);
            if (unit.getPlayer() != n || unitActionAssignment != null) continue;
            this.masks[n][unit.getY()][unit.getX()][0] = 1;
            this.getValidActionArray(unit, unitTypeTable, this.masks[n][unit.getY()][unit.getX()], n2, 1);
        }
        return this.masks[n];
    }

    public byte[] getBinaryMask(int n) {
        UnitTypeTable unitTypeTable = this.gs.getUnitTypeTable();
        PhysicalGameState physicalGameState = this.gs.getPhysicalGameState();
        List list = physicalGameState.getUnits().stream().filter(unit -> unit.getPlayer() == n && this.gs.getActionAssignment(unit) == null).collect(Collectors.toList());
        int n2 = unitTypeTable.getMaxAttackRange() * 2 + 1;
        int n3 = 22 + unitTypeTable.getUnitTypes().size() + n2 * n2;
        byte[] byArray = new byte[list.size() * (2 + n3)];
        int[] nArray = new int[n3];
        for (int i = 0; i < list.size(); ++i) {
            Unit unit2 = (Unit)list.get(i);
            int n4 = i * (2 + n3);
            byArray[n4 + 0] = (byte)unit2.getY();
            byArray[n4 + 1] = (byte)unit2.getX();
            this.getValidActionArray(unit2, unitTypeTable, nArray, n2, 0);
            if (this.debugLevel > 0) {
                int[] nArray2 = new int[n3];
                UnitAction.getValidActionArray((Unit)unit2, (GameState)this.gs, (UnitTypeTable)unitTypeTable, (int[])nArray2, (int)n2, (int)0);
                for (int j = 0; j < nArray.length; ++j) {
                    if (nArray[j] == 0 || nArray2[j] != 0) continue;
                    System.err.println("Action mask for unit " + String.valueOf(unit2) + " is true for index " + j + " despite base mask not.");
                }
            }
            for (int j = 0; j < nArray.length; ++j) {
                byArray[n4 + 2 + j] = (byte)nArray[j];
            }
            Arrays.fill(nArray, 0);
        }
        return byArray;
    }

    public void getValidActionArray(Unit unit, UnitTypeTable unitTypeTable, int[] nArray, int n, int n2) {
        List<UnitAction> list = this.getUnitActions(unit);
        int n3 = n / 2;
        int n4 = unitTypeTable.getUnitTypes().size();
        for (UnitAction unitAction : list) {
            nArray[n2 + unitAction.getType()] = 1;
            switch (unitAction.getType()) {
                case 0: {
                    break;
                }
                case 1: {
                    nArray[n2 + 6 + unitAction.getDirection()] = 1;
                    break;
                }
                case 2: {
                    nArray[n2 + 6 + 4 + unitAction.getDirection()] = 1;
                    break;
                }
                case 3: {
                    nArray[n2 + 6 + 4 + 4 + unitAction.getDirection()] = 1;
                    break;
                }
                case 4: {
                    nArray[n2 + 6 + 4 + 4 + 4 + unitAction.getDirection()] = 1;
                    nArray[n2 + 6 + 4 + 4 + 4 + 4 + unitAction.getUnitType().ID] = 1;
                    break;
                }
                case 5: {
                    int n5 = unitAction.getLocationX() - unit.getX();
                    int n6 = unitAction.getLocationY() - unit.getY();
                    nArray[n2 + 6 + 4 + 4 + 4 + 4 + n4 + (n3 + n6) * n + (n3 + n5)] = 1;
                    break;
                }
            }
        }
    }

    public List<UnitAction> getUnitActions(Unit unit) {
        int n;
        int n2;
        ArrayList<UnitAction> arrayList = new ArrayList<UnitAction>();
        PhysicalGameState physicalGameState = this.gs.getPhysicalGameState();
        int n3 = unit.getPlayer();
        Player player = physicalGameState.getPlayer(n3);
        int n4 = unit.getX();
        int n5 = unit.getY();
        Unit unit2 = null;
        Unit unit3 = null;
        Unit unit4 = null;
        Unit unit5 = null;
        UnitActionAssignment unitActionAssignment = null;
        UnitActionAssignment unitActionAssignment2 = null;
        UnitActionAssignment unitActionAssignment3 = null;
        UnitActionAssignment unitActionAssignment4 = null;
        for (Unit unit6 : physicalGameState.getUnits()) {
            if (unit6.getX() == n4) {
                if (unit6.getY() == n5 - 1) {
                    unit2 = unit6;
                    unitActionAssignment = this.gs.getActionAssignment(unit2);
                    continue;
                }
                if (unit6.getY() != n5 + 1) continue;
                unit4 = unit6;
                unitActionAssignment3 = this.gs.getActionAssignment(unit4);
                continue;
            }
            if (unit6.getY() != n5) continue;
            if (unit6.getX() == n4 - 1) {
                unit5 = unit6;
                unitActionAssignment4 = this.gs.getActionAssignment(unit5);
                continue;
            }
            if (unit6.getX() != n4 + 1) continue;
            unit3 = unit6;
            unitActionAssignment2 = this.gs.getActionAssignment(unit3);
        }
        UnitType unitType = unit.getType();
        if (unitType.canAttack) {
            int n6 = unitType.attackTime;
            if (unitType.attackRange == 1) {
                if (n5 > 0 && unit2 != null && unit2.getPlayer() != n3 && unit2.getPlayer() >= 0 && (this.allowMoveAwayAttacks || !this.isUnitMovingWithinTimesteps(unitActionAssignment, n6))) {
                    arrayList.add(new UnitAction(5, unit2.getX(), unit2.getY()));
                }
                if (n4 < physicalGameState.getWidth() - 1 && unit3 != null && unit3.getPlayer() != n3 && unit3.getPlayer() >= 0 && (this.allowMoveAwayAttacks || !this.isUnitMovingWithinTimesteps(unitActionAssignment2, n6))) {
                    arrayList.add(new UnitAction(5, unit3.getX(), unit3.getY()));
                }
                if (n5 < physicalGameState.getHeight() - 1 && unit4 != null && unit4.getPlayer() != n3 && unit4.getPlayer() >= 0 && (this.allowMoveAwayAttacks || !this.isUnitMovingWithinTimesteps(unitActionAssignment3, n6))) {
                    arrayList.add(new UnitAction(5, unit4.getX(), unit4.getY()));
                }
                if (n4 > 0 && unit5 != null && unit5.getPlayer() != n3 && unit5.getPlayer() >= 0 && (this.allowMoveAwayAttacks || !this.isUnitMovingWithinTimesteps(unitActionAssignment4, n6))) {
                    arrayList.add(new UnitAction(5, unit5.getX(), unit5.getY()));
                }
            } else {
                int n7 = unitType.attackRange * unitType.attackRange;
                for (Unit unit7 : physicalGameState.getUnits()) {
                    if (unit7.getPlayer() < 0 || unit7.getPlayer() == n3 || (n2 = (unit7.getX() - n4) * (unit7.getX() - n4)) + (n = (unit7.getY() - n5) * (unit7.getY() - n5)) > n7) continue;
                    UnitActionAssignment unitActionAssignment5 = this.gs.getActionAssignment(unit7);
                    if (!this.allowMoveAwayAttacks && this.isUnitMovingWithinTimesteps(unitActionAssignment5, n6)) continue;
                    arrayList.add(new UnitAction(5, unit7.getX(), unit7.getY()));
                }
            }
        }
        int n8 = unit.getResources();
        if (unitType.canHarvest) {
            if (n8 == 0) {
                if (n5 > 0 && unit2 != null && unit2.getType().isResource) {
                    arrayList.add(new UnitAction(2, 0));
                }
                if (n4 < physicalGameState.getWidth() - 1 && unit3 != null && unit3.getType().isResource) {
                    arrayList.add(new UnitAction(2, 1));
                }
                if (n5 < physicalGameState.getHeight() - 1 && unit4 != null && unit4.getType().isResource) {
                    arrayList.add(new UnitAction(2, 2));
                }
                if (n4 > 0 && unit5 != null && unit5.getType().isResource) {
                    arrayList.add(new UnitAction(2, 3));
                }
            }
            if (n8 > 0) {
                if (n5 > 0 && unit2 != null && unit2.getType().isStockpile && unit2.getPlayer() == n3) {
                    arrayList.add(new UnitAction(3, 0));
                }
                if (n4 < physicalGameState.getWidth() - 1 && unit3 != null && unit3.getType().isStockpile && unit3.getPlayer() == n3) {
                    arrayList.add(new UnitAction(3, 1));
                }
                if (n5 < physicalGameState.getHeight() - 1 && unit4 != null && unit4.getType().isStockpile && unit4.getPlayer() == n3) {
                    arrayList.add(new UnitAction(3, 2));
                }
                if (n4 > 0 && unit5 != null && unit5.getType().isStockpile && unit5.getPlayer() == n3) {
                    arrayList.add(new UnitAction(3, 3));
                }
            }
        }
        for (UnitType unitType2 : unitType.produces) {
            UnitAction unitAction;
            int n9;
            if (player.getResources() < unitType2.cost + this.base_ru.getResourcesUsed(n3)) continue;
            int n10 = n5 > 0 ? physicalGameState.getTerrain(n4, n5 - 1) : 1;
            n2 = n4 < physicalGameState.getWidth() - 1 ? physicalGameState.getTerrain(n4 + 1, n5) : 1;
            n = n5 < physicalGameState.getHeight() - 1 ? physicalGameState.getTerrain(n4, n5 + 1) : 1;
            int n11 = n9 = n4 > 0 ? physicalGameState.getTerrain(n4 - 1, n5) : 1;
            if (n10 == 0 && physicalGameState.getUnitAt(n4, n5 - 1) == null && (unitAction = new UnitAction(4, 0, unitType2)).resourceUsage(unit, physicalGameState).consistentWith(this.base_ru, this.gs)) {
                arrayList.add(unitAction);
            }
            if (n2 == 0 && physicalGameState.getUnitAt(n4 + 1, n5) == null && (unitAction = new UnitAction(4, 1, unitType2)).resourceUsage(unit, physicalGameState).consistentWith(this.base_ru, this.gs)) {
                arrayList.add(unitAction);
            }
            if (n == 0 && physicalGameState.getUnitAt(n4, n5 + 1) == null && (unitAction = new UnitAction(4, 2, unitType2)).resourceUsage(unit, physicalGameState).consistentWith(this.base_ru, this.gs)) {
                arrayList.add(unitAction);
            }
            if (n9 != 0 || physicalGameState.getUnitAt(n4 - 1, n5) != null || !(unitAction = new UnitAction(4, 3, unitType2)).resourceUsage(unit, physicalGameState).consistentWith(this.base_ru, this.gs)) continue;
            arrayList.add(unitAction);
        }
        if (unitType.canMove) {
            UnitAction unitAction;
            UnitAction unitAction2;
            UnitAction unitAction3;
            UnitAction unitAction4;
            int n12 = n5 > 0 ? physicalGameState.getTerrain(n4, n5 - 1) : 1;
            int n13 = n4 < physicalGameState.getWidth() - 1 ? physicalGameState.getTerrain(n4 + 1, n5) : 1;
            int n14 = n5 < physicalGameState.getHeight() - 1 ? physicalGameState.getTerrain(n4, n5 + 1) : 1;
            int n15 = n2 = n4 > 0 ? physicalGameState.getTerrain(n4 - 1, n5) : 1;
            if (n12 == 0 && unit2 == null && (unitAction4 = new UnitAction(1, 0)).resourceUsage(unit, physicalGameState).consistentWith(this.base_ru, this.gs)) {
                arrayList.add(unitAction4);
            }
            if (n13 == 0 && unit3 == null && (unitAction3 = new UnitAction(1, 1)).resourceUsage(unit, physicalGameState).consistentWith(this.base_ru, this.gs)) {
                arrayList.add(unitAction3);
            }
            if (n14 == 0 && unit4 == null && (unitAction2 = new UnitAction(1, 2)).resourceUsage(unit, physicalGameState).consistentWith(this.base_ru, this.gs)) {
                arrayList.add(unitAction2);
            }
            if (n2 == 0 && unit5 == null && (unitAction = new UnitAction(1, 3)).resourceUsage(unit, physicalGameState).consistentWith(this.base_ru, this.gs)) {
                arrayList.add(unitAction);
            }
        }
        arrayList.add(new UnitAction(0, 1));
        return arrayList;
    }

    public boolean isUnitMovingWithinTimesteps(UnitActionAssignment unitActionAssignment, int n) {
        if (unitActionAssignment == null || unitActionAssignment.action.getType() != 1) {
            return false;
        }
        int n2 = unitActionAssignment.time + unitActionAssignment.action.ETA(unitActionAssignment.unit) - this.gs.getTime();
        return n2 < n;
    }

    public static byte byteClampValue(int n) {
        return (byte)(Math.max(0, Math.min(n, 255)) - 128);
    }

    public byte[] getPlayerResources(int n) {
        return new byte[]{(byte)this.gs.getPlayer(n).getResources(), (byte)this.gs.getPlayer(1 - n).getResources()};
    }

    public static int[][] toVectorAction(GameState gameState, PlayerAction playerAction) {
        UnitTypeTable unitTypeTable = gameState.getUnitTypeTable();
        int n = unitTypeTable.getMaxAttackRange();
        int n2 = unitTypeTable.getMaxAttackRange() * 2 + 1;
        int n3 = gameState.getPhysicalGameState().getWidth();
        ArrayList<int[]> arrayList = new ArrayList<int[]>();
        for (Pair pair : playerAction.getActions()) {
            Unit unit = (Unit)pair.m_a;
            UnitAction unitAction = (UnitAction)pair.m_b;
            if (!gameState.getActionAssignment((Unit)unit).action.equals((Object)unitAction)) {
                System.out.println(String.valueOf(unit) + " hasn't been issued " + String.valueOf(unitAction) + ". Skipping.");
                continue;
            }
            int[] nArray = new int[8];
            nArray[0] = unit.getX() + unit.getY() * n3;
            nArray[1] = unitAction.getType();
            switch (unitAction.getType()) {
                case 0: {
                    break;
                }
                case 1: {
                    nArray[2] = unitAction.getDirection();
                    break;
                }
                case 2: {
                    nArray[3] = unitAction.getDirection();
                    break;
                }
                case 3: {
                    nArray[4] = unitAction.getDirection();
                    break;
                }
                case 4: {
                    nArray[5] = unitAction.getDirection();
                    nArray[6] = unitAction.getUnitType().ID;
                    break;
                }
                case 5: {
                    int n4 = unitAction.getLocationX() - unit.getX();
                    int n5 = unitAction.getLocationY() - unit.getY();
                    nArray[7] = n4 + n + (n5 + n) * n2;
                    break;
                }
            }
            arrayList.add(nArray);
        }
        return (int[][])arrayList.toArray((T[])new int[0][]);
    }
}

