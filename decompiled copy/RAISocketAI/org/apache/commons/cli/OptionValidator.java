/*
 * Decompiled with CFR 0.152.
 */
package org.apache.commons.cli;

final class OptionValidator {
    OptionValidator() {
    }

    private static boolean isValidChar(char c) {
        return Character.isJavaIdentifierPart(c);
    }

    private static boolean isValidOpt(char c) {
        return OptionValidator.isValidChar(c) || c == '?' || c == '@';
    }

    static String validate(String option) throws IllegalArgumentException {
        if (option == null) {
            return null;
        }
        if (option.length() == 1) {
            char ch = option.charAt(0);
            if (!OptionValidator.isValidOpt(ch)) {
                throw new IllegalArgumentException("Illegal option name '" + ch + "'");
            }
        } else {
            for (char ch : option.toCharArray()) {
                if (OptionValidator.isValidChar(ch)) continue;
                throw new IllegalArgumentException("The option '" + option + "' contains an illegal character : '" + ch + "'");
            }
        }
        return option;
    }
}

