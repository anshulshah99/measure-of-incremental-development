import os
import pathlib
import tokenize
import io
from collections import Counter
import difflib as dl
import pandas as pd
from datetime import datetime, timedelta
import csv
import math
import random
import pickle
import time
from PIL import Image, ImageDraw, ImageFont
from radon.metrics import h_visit, h_visit_ast, mi_parameters
from radon.visitors import HalsteadVisitor
from radon.raw import analyze
import ast
import matplotlib.pyplot as plt

def get_tokens(token_list, type = True, line_bool = True, start = True, string = True):
    lst = []
    for t in token_list:
        vals = []
        if type:
            vals.append(t.type)
        if string:
            vals.append(t.string)
        if start:
            vals.append(t.start)
        if line_bool:
            vals.append(t.line)
        lst.append(tuple(vals))
    return lst

def token_counter(tok):
    tokens = []
    isIndent = False
    try:
        for token in tokenize.tokenize(io.BytesIO(tok.encode('utf-8')).readline):
            if token.type not in [0, 4, 6, 53, 56, 60, 61, 62]:
                tokens.append(token)
            if token.type == 5:
                isIndent = True
    except tokenize.TokenError:
        pass
    return tokens, isIndent


def is_def(line):
    # Check if a line is a function definition, such as 'def rainfall(rain_list):'
    if "def" in line and "(" in line and ")" in line:
        return True
    return False

def remove_comments(snapshot):
    new_snap = []
    for line in snapshot:
        if "#" in line:
            idx = line.index("#")
            line = line[:idx]
        new_snap.append(line)
    return new_snap

def get_adjustment_location(forward_prog_steps, deleted_line, adjustment_locations):
    found = False
    for k in range(len(forward_prog_steps)):
        fp_step = forward_prog_steps[k]
        for i in range(len(fp_step)):
            if deleted_line.strip() == fp_step[i].strip():
                adjustment_locations.add(k)
                found = True
                fp_step[i] = deleted_line.strip()
    if not found:
        adjustment_locations.add(len(forward_prog_steps) - 1)
    return forward_prog_steps, adjustment_locations


def getDiffBetweenRuns(run1, run2):
    list_of_diffs = []
    for diff in dl.ndiff(run1, run2):
        list_of_diffs.append(diff)
    return list_of_diffs


def getTokensChanged(addedLine, deletedLine):
    tokens_in_add_line, isIndentAdded = token_counter(addedLine)
    tokens_in_del_line, isIndentDeleted = token_counter(deletedLine)

    processed_add_tokens = get_tokens(tokens_in_add_line, line_bool=False, start=False)
    processed_del_tokens = get_tokens(tokens_in_del_line, line_bool=False, start=False)

    tokensAdded = list((Counter(processed_add_tokens) - Counter(processed_del_tokens)).elements())
    tokensDeleted = list((Counter(processed_del_tokens) - Counter(processed_add_tokens)).elements())
    return tokensAdded, tokensDeleted, isIndentAdded, isIndentDeleted

def checkRearrangedLines(numLinesAdded, numLinesDeleted, linesAdded, linesDeleted):
    rearranged = False
    matches = 0
    if (numLinesAdded == numLinesDeleted) and numLinesAdded > 0:
        for l in linesAdded:
            if l in linesDeleted:
                matches += 1
        if matches == numLinesAdded:
            rearranged = True
    return rearranged


def classify_snapshots(runs):
    label_list = []
    progress_code = []
    forward_prog_lst = []
    adjust_locs = []
    for i in range(len(runs) - 1):
        tokens_added = 0
        tokens_deleted = 0
        lines_added = 0
        lines_deleted = 0
        print_statements = 0
        changed_out_funcs = 0
        edited_tokens = 0
        lines_added_lst = []
        lines_deleted_lst = []
        adjustment_locations = set()

        first_run = remove_comments(runs[i])
        second_run = remove_comments(runs[i + 1])

        list_of_diffs = getDiffBetweenRuns(first_run, second_run)

        lines_checked_tracker = []
        for j in range(len(list_of_diffs)):
            line = list_of_diffs[j].strip()
            if line.startswith("?"):
                if list_of_diffs[j + 1].startswith("+") and list_of_diffs[j - 1].startswith("-"):
                    indexAddedLine = j+1
                    indexDeletedLine = j-1
                    # if the question mark is between the - and +

                elif list_of_diffs[j - 1].startswith("+") and list_of_diffs[j - 2].startswith("-"):
                    indexAddedLine = j - 1
                    indexDeletedLine = j - 2
                    # if the question mark is below the - and +
                else:
                    continue
                added_line = list_of_diffs[indexAddedLine][2:]
                deleted_line = list_of_diffs[indexDeletedLine][2:]
                lines_checked_tracker.append(indexAddedLine)
                lines_checked_tracker.append(indexDeletedLine)


                numTokensAdded, numTokensDeleted, isIndentAdded, isIndentDeleted = getTokensChanged(added_line, deleted_line)

                if "print" in added_line or "print" in deleted_line:
                    print_statements += 1
                elif (not isIndentAdded and not is_def(added_line)):
                    changed_out_funcs += (len(numTokensAdded))
                else:  # (indent or is_def(added_line)):
                    progress_code, adjustment_locations = get_adjustment_location(progress_code, deleted_line,
                                                                                  adjustment_locations)
                    edited_tokens += (len(numTokensAdded) + len(numTokensDeleted))

        # Once these lines have been checked, process the lines that are completely new or completely deleted
        for j in range(len(list_of_diffs)):
            line = list_of_diffs[j].strip()
            if (line.startswith("+") or line.startswith("-")) and j not in lines_checked_tracker:
                action = line[0]
                rest = line[2:]
                if len(rest) == 0:
                    continue
                if action == "-":
                    progress_code, adjustment_locations = get_adjustment_location(progress_code, rest,
                                                                                  adjustment_locations)
                    list_deleted_toks, indent = token_counter(rest)
                    if (indent or is_def(rest)) and "print" not in rest:
                        lines_deleted_lst.append(rest)
                        lines_deleted += 1
                    else:
                        changed_out_funcs += len(list_deleted_toks)
                if action == "+":
                    list_added_toks, indent = token_counter(rest)
                    if (indent or is_def(rest))and "print" not in rest:
                        lines_added += 1
                        lines_added_lst.append(rest)
                    else:
                        changed_out_funcs += len(list_added_toks)

        label = ""
        rearranged = checkRearrangedLines(lines_added, lines_deleted, lines_added_lst, lines_deleted_lst)

        if changed_out_funcs > 0 or print_statements > 0:
            label = "TEST"
        if lines_added == 0 and lines_deleted == 0 and print_statements == 0 and changed_out_funcs == 0 and edited_tokens == 0 and not rearranged:
            label = "NONE"
        if rearranged or (edited_tokens > 0 and lines_added == 0) or (lines_deleted > 0 and lines_added == 0):
            label = "ADJUSTMENT"
        if lines_added > 0 and print_statements < lines_added and not rearranged:
            # print(lines_added_lst)
            if label == "ADJUSTMENT":
                label_list.append(label)
                adjust_locs.append(adjustment_locations)
            label = "FORWARD_PROG:{}".format(lines_added)
            progress_code.append(lines_added_lst)
            forward_prog_lst.append(lines_added_lst)
        if label == "ADJUSTMENT":
            adjust_locs.append(adjustment_locations)
        label_list.append(label)
    return label_list, forward_prog_lst, adjust_locs

