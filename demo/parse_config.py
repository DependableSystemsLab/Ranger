import yaml
import sys

try:
	configFile = str(sys.argv[1])
except:
	print("no yaml file specified, will use the default one")

def yamlFaultParams(pStream):
	"Read fault params from YAML file"
	# NOTE: We assume pStream is a valid YAML stream
	params = yaml.load(pStream)
	return params

def configFaultParams(paramFile = None):
	"Return the fault params from different files"
	if paramFile == None:
		return staticFaultParams()

	params = {}
	try:
		paramStream = open(paramFile, "r")
	except IOError:
		print "Unable to open file ", paramFile
		return params

	# Check if the file extension is .yaml, and if so parse the Stream 
	# (right now, this is the only supported format, but can be extended)
	if paramFile.endswith(".yaml"):
		params = yamlFaultParams(paramStream)
	else:
		print "Unknown file format: ", paramFile
		
	#print params
	return params
 
try:
	params = configFaultParams(configFile)
except:
	params = configFaultParams( "default.yaml" ) 


lines = [] 
lines.append( 'act = \'{}\''.format(params['act']) )
lines.append( 'op_follow_act = {}'.format(params['op_follow_act']) )
lines.append( 'special_op_follow_act = \'{}\''.format(params['special_op_follow_act']) )
lines.append( 'up_bound = {}'.format(params['up_bound']) )
lines.append( 'low_bound = {}'.format(params['low_bound']) )
lines.append( 'from ranger import * ' )
lines.append( 'OLD_SESS, {}, dup_cnt, dummy_graph_dup_cnt = insertRanger({}, act, op_follow_act, special_op_follow_act, up_bound, low_bound ) '.format(params['org_sess_name'], params['org_sess_name']) )

for each in params['op_in_new_graph']:
	lines.append( '{} = get_op_from_new_graph(sess ={}, op = {}, graph_dup_cnt= dup_cnt, dummy_graph_dup_cnt = dummy_graph_dup_cnt)'.format( each, params['org_sess_name'], each ) )

codeFile_to_be_written = "ranger-code-to-insert.py"

a = open(codeFile_to_be_written, "w")

for each in lines:
	a.write(each + "\n")









