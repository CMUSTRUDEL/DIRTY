from collections import defaultdict
from util import UNDEF_ADDR, CFuncGraph, GraphBuilder, hexrays_vars, get_expr_name
import idaapi
import ida_hexrays
import ida_kernwin
import ida_pro
import json
import jsonlines
import os
import re
import subprocess
import cStringIO

# Dictionary mapping variable ids to (orig, orig) pairs
varnames = dict()
var_id = 0
sentinel_vars = re.compile('@@VAR_[0-9]+')

dire_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
RUN_ONE = os.path.join(dire_dir, "run_one.py")
MODEL = os.path.join(dire_dir, 'data', 'saved_models', 'model.hybrid.bin')

class RenamedGraphBuilder(GraphBuilder):
    def __init__(self, cg, func):
        self.func = func
        super(RenamedGraphBuilder, self).__init__(cg)

    def visit_expr(self, e):
        global var_id
        if e.op is ida_hexrays.cot_var:
            # Save original name of variable
            original_name = get_expr_name(e)
            if not sentinel_vars.match(original_name):
                # Save names
                varnames[var_id] = (original_name, original_name)
                # Rename variables to @@VAR_[id]@@[orig name]@@[orig name]
                self.func.get_lvars()[e.v.idx].name = \
                    '@@VAR_' + str(var_id) + '@@' + original_name + '@@' + original_name
                var_id += 1
        return self.process(e)

# Process a single function given its EA
def func(ea):
    f = idaapi.get_func(ea)
    function_name = GetFunctionName(ea)
    if f is None:
        print('Please position the cursor within a function')

    cfunc = None
    try:
        cfunc = idaapi.decompile(f)
    except ida_hexrays.DecompilationFailure as e:
        print('Failed to decompile %x: %s!' % (ea, function_name))
        raise e

    # Rename decompilation graph
    cg = CFuncGraph(None)
    gb = GraphBuilder(cg)
    gb.apply_to(cfunc.body, None)
    #ac = AddressCollector(cg)
    #ac.collect()
    rg = RenamedGraphBuilder(cg, cfunc)
    rg.apply_to(cfunc.body, None)

    # Create tree from collected names
    cfunc.build_c_tree()
    new_graph = CFuncGraph(None)
    new_builder = GraphBuilder(new_graph)
    new_builder.apply_to(cfunc.body, None)
    function_info = dict()
    function_info["function"] = function_name
    function_info["ast"] = new_graph.json_tree(0)
    raw_code = ""
    for line in cfunc.get_pseudocode():
        raw_code += idaapi.tag_remove(line.line) + '\n'
    function_info["raw_code"] = raw_code
    return function_info

class custom_action_handler(ida_kernwin.action_handler_t):
    def __init__(self):
        ida_kernwin.action_handler_t.__init__(self)

class collect_vars(custom_action_handler):
    def activate(self, ctx):
        ea = ScreenEA()
        if ea is None:
            warning("Current function not found.")
        else:
            f = cStringIO.StringIO()
            with jsonlines.Writer(f) as writer:
                try:
                    info = func(ea)
                    # We must set the working directory to the dire dir to open the model correctly
                    os.chdir(dire_dir)
                    p = subprocess.Popen([RUN_ONE, '--model', MODEL], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
                    #print(info)
                    writer.write(info)
                    json_results = p.communicate(input=f.getvalue())[0].decode()
                    print(json_results)
                    results = json.loads(json_results)
                    best_results = results[0][0]
                    #print("best: ", best_results)
                    tuples = map(lambda x: (x[0], x[1]['new_name']), best_results.items())
                    for x, y in tuples:
                        print('I should rename %s to %s'%(x,y))
                    #print(tuples)
                    #writer.write(func(ea))
                except ida_hexrays.DecompilationFailure:
                    warning("Decompilation failed")
        return 1

def main():
    # Collect decompilation info
    cv = collect_vars()
    cv.activate(None)

idaapi.autoWait()
if not idaapi.init_hexrays_plugin():
    idaapi.load_plugin('hexrays')
    idaapi.load_plugin('hexx64')
    if not idaapi.init_hexrays_plugin():
        print('Unable to load Hex-rays')
    else:
        print('Hex-rays version %s has been detected' % idaapi.get_hexrays_version())
main()
