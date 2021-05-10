import yaml
import sys

try:
	configFile = str(sys.argv[1])
except:
	print("no yaml file specified, will use the default one")

try:
	outputFile = str(sys.argv[2])
except:
	print("no output yaml file specified, will use the default one")


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
	params = configFaultParams( "profiling.yaml" ) 
	print('use the default config file: profiling.yaml' )




lines = []
lines.append( 'percentage_of_img = {}'.format( params['percentage_of_img']) )
lines.append( 'act = \'{}\''.format(params['act']) )
lines.append( 'save_raw_data = {}'.format(params['save_raw_data']) )
lines.append( 'restriction_bound_pert = {}'.format(params['restriction_val']) )
lines.append( 'sess = {}'.format(params['sess_name']) )
lines.append( 'x_data = {}'.format(params['x_data']) )


template_file = "profiling_template.py"

with open(template_file, 'r+') as f:
	content = f.read()

with open('profiling_filled_template.py', 'w') as f:
	# assign sess.run() with the feed dict 
	num_of_feeddict = len(params['feed_dict']) 
	string = 'value = sess.run(ACT_op, feed_dict={'
	for i in range(num_of_feeddict):
		tmp = params['feed_dict'][i].items()[0]
		string += "{} : {},".format(tmp[0], tmp[1])

	string = string.rstrip(",")
	string += "})"

	string = string.replace( params['feed_dict'][0].items()[0][1], 'img' ) # fill the feed_dict input variable 

	content = content.replace("value = sess.run()", string) 

	content = content.replace( 'outputFile=' , 'outputFile=\'{}\''.format(outputFile) )

	f.seek(0, 0)
	for each in lines:
		f.write(each + "\n")

	f.write(content)











