/*
 * Decompiled with CFR 0.152.
 */
package org.apache.commons.cli;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public interface CommandLineParser {
    public CommandLine parse(Options var1, String[] var2) throws ParseException;

    public CommandLine parse(Options var1, String[] var2, boolean var3) throws ParseException;
}

