from ranges import merge, contains
def _bad(s):
    try: merge([s])
    except ValueError: return True
    return False
def _t1(): return merge(["[1,3)", "[3,5]"]) == ["[1,5]"] and merge(["[1,3]", "(3,5]"]) == ["[1,5]"]
def _t2(): return merge(["[1,3)", "(3,5]"]) == ["[1,3)", "(3,5]"]
def _t3(): return merge(["[3,3)", "(4,4]", "(5,5)"]) == [] and merge(["[3,3]"]) == ["[3,3]"]
def _t4(): return merge(["[3,3]", "(3,6)"]) == ["[3,6)"] and merge(["(1,3)", "[3,3]"]) == ["(1,3]"]
def _t5(): return merge(["[1,5)", "[2,5]"]) == ["[1,5]"] and merge(["(1,4)", "[1,2]"]) == ["[1,4)"]
def _t6(): return merge(["[10,12]", "[ -3 , -1 )"]) == ["[-3,-1)", "[10,12]"]
def _t7(): return merge(["[1,10]", "[2,3)", "(4,5)"]) == ["[1,10]"] and merge(["[1,2)", "(2,3)", "[2,2]"]) == ["[1,3)"]
def _t8(): return all(_bad(s) for s in ["[5,1]", "1,2", "[a,2]", "{1,2}", "[1.5,2]", "[1;2]"])
def _t9(): return contains(["[1,3)"], 3) is False and contains(["[1,3]"], 3) is True and contains(["(1,3)", "[3,4]"], 3) is True and contains(["(1,3)"], 1) is False
def _t10():
    x = ["[5,6]", "[1,2]"]; merge(x); return x == ["[5,6]", "[1,2]"] and merge(["(2,4)", "(1,3)"]) == ["(1,4)"]
C = [_t1, _t2, _t3, _t4, _t5, _t6, _t7, _t8, _t9, _t10]
