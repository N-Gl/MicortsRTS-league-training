/*
 * Decompiled with CFR 0.152.
 */
package org.apache.commons.cli;

import org.apache.commons.cli.Options;
import org.apache.commons.cli.Parser;

@Deprecated
public class BasicParser
extends Parser {
    @Override
    protected String[] flatten(Options options, String[] arguments, boolean stopAtNonOption) {
        return arguments;
    }
}

