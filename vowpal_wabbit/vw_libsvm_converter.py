'''
python vw_libsvm_converter.py -i Eval.svm -o Eval.svm.vw -b 
'''
import sys
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-i", "--input-file", action="store", type='string', dest="inputFile", help="Input file")
parser.add_option("-o", "--output-file", action="store", type='string', dest="outputFile", help="Output file")
parser.add_option("-b", "--is-binary", action="store_true", dest="isBinary")
settings, args = parser.parse_args(sys.argv)

#########################

f = open(settings.inputFile, 'rb')
g = open(settings.outputFile, 'wb')
for i, line in enumerate(f):
	if i % 10000 == 0:
		print "[idx]: %d" % i
	if not line:
		continue
	line = line.replace('\n', '')
	parts = line.split()
	if settings.isBinary and parts[0] == "0":
		parts[0] = "-1"
	out = parts[0] + " | " + " ".join(parts[1:])
	g.write(out)
	g.write("\n")

# done
