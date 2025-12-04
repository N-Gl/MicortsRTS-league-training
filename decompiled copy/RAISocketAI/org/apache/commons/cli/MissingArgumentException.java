/*
 * Decompiled with CFR 0.152.
 */
package org.apache.commons.cli;

import org.apache.commons.cli.Option;
import org.apache.commons.cli.ParseException;

public class MissingArgumentException
extends ParseException {
    private static final long serialVersionUID = -7098538588704965017L;
    private Option option;

    public MissingArgumentException(Option option) {
        this("Missing argument for option: " + option.getKey());
        this.option = option;
    }

    public MissingArgumentException(String message) {
        super(message);
    }

    public Option getOption() {
        return this.option;
    }
}

