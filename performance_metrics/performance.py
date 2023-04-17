import numpy
import os
import sys


def seqste(d1, *more):
	"""
	The name "seqste" may come from the fact that the function is used for sequential data analysis and estimation of 
	standard error (abbreviated as "ste"). The name also refers to the article by Baker and Nissim, which introduces expressions for 
	combining standard errors of two groups and for sequential standard error in Nature in 1963.
	d1 and d2 both dicts with entries 'mean', 'ste' and 'n'.
	Return a similar dict with a combined estimate of the mean and standard-error-of-mean.
		
	Baker, R.W.R & Nissim, J.A. (1963):
		Expressions for Combining Standard Errors of Two Groups and for Sequential Standard Error
		Nature 198, 1020; doi:10.1038/1981020a0
		http://www.nature.com/nature/journal/v198/n4884/abs/1981020a0.html
	"""
	if len(more) == 0: return d1
	d2 = more[0]
	keys = ['mean','ste','n']
	if sorted(d1.keys()) != sorted(keys) or sorted(d2.keys()) != sorted(keys):
		raise ValueError('data inputs should be dicts with fields %s' % ','.join(keys))
	def conv(x):
		if isinstance(x, numpy.ndarray): x = numpy.asarray(x, dtype=numpy.float64)
		if isinstance(x, (int,bool)): x = float(x)
		return x
	m1,e1,n1 = [conv(d1[k]) for k in keys]
	m2,e2,n2 = [conv(d2[k]) for k in keys]
	n3 = n1 + n2
	v3 = n1*(n1-1)*e1**2 + n2*(n2-1)*e2**2 + n1*n2*(m1-m2)**2/n3
	v3 /= n3 * (n3-1)
	e3 = v3 ** 0.5
	m3 = (m1*n1 + m2*n2) / n3
	result = {'mean':m3,'ste':e3,'n':int(n3)}
	return seqste(result, *more[1:])

def confuse(true, predicted, labels=None, exemplar_dim=0):
	"""
	Returns a confusion matrix and a list of unique labels, based on paired lists of
	true and predicted labels.
	
	Output rows correspond to the possible true labels and columns correspond to the
	possible predicted labels. This is the ordering assumed in, for example,
	balanced_loss().
	"""###
	true = numpy.asarray(true)
	predicted = numpy.asarray(predicted)
	nd = max(exemplar_dim+1, len(true.shape), len(predicted.shape))
	if exemplar_dim < 0: exemplar_dim += nd
	true = true.swapaxes(exemplar_dim, 0)
	predicted = predicted.swapaxes(exemplar_dim, 0)
	if len(true) != len(predicted): raise ValueError('mismatched numbers of true and predicted labels')

	def isequal(a,b):
		if isinstance(a, str) and isinstance(b, str): return a == b											# "basestring" is not defined in Python 3, changed to "str"
		a = numpy.asarray(a)
		b = numpy.asarray(b)
		ndd = len(b.shape) - len(a.shape)
		if ndd > 0: a.shape += (1,) * ndd
		if ndd < 0: b.shape += (1,) * -ndd
		if a.shape != b.shape: return False
		return numpy.logical_or(a == b, numpy.logical_and(numpy.isnan(a), numpy.isnan(b))).all()
	def find(a, b, append=False):
		for i in range(len(b)):
			if isequal(a, b[i]): return i
		if append: b.append(a); return len(b)-1
		else: return None

	n = len(true)
	c = {}
	found_labels = []
	for i in range(n):
		tv,pv = true[i],predicted[i]
		ti = find(tv, found_labels, append=True)
		pi = find(pv, found_labels, append=True)
		key = (ti,pi)
		c[key] = c.get(key,0) + 1

	if labels is None:
		labels = list(found_labels)
		try: labels.sort()
		except: pass
	else:
		labels = list(labels)
		for fi in found_labels:
			if find(fi, labels) is None: raise ValueError('inputs contain labels not listed in the <labels> argument')
	nclasses = len(labels)
	C = numpy.zeros((nclasses, nclasses), dtype=numpy.float64)
	for i in range(nclasses):
		ti = find(labels[i], found_labels, append=False)
		if ti is None: continue
		for j in range(nclasses):
			tj = find(labels[j], found_labels, append=False)
			if tj is None: continue
			C[i,j] = c.get((ti,tj), 0)
	return C,labels
	
def balanced_loss(true=None, predicted=None, confusion_matrix=None):
	"""
	err, se = balanced_loss(true, predicted)
	err, se = balanced_loss(confusion_matrix=C)
		where C = confuse(true, predicted)
	
	A classification loss function. As in confuse(), each row of <true> or
	<predicted> is a label for one instance or data point.
	
	balanced_loss() is asymmetric with regard to its inputs: it is the mean
	of the misclassification rates on each of the classes (as per <true>).
	"""###	
	if confusion_matrix is None:
		predicted = numpy.asarray(predicted).flatten()
		if (predicted > numpy.floor(predicted)).any(): predicted = numpy.sign(predicted)
		confusion_matrix,labels = confuse(true=true, predicted=predicted)

	confusion_matrix = numpy.asarray(confusion_matrix, dtype=numpy.float64)
	hits = confusion_matrix.diagonal()
	totals = confusion_matrix.sum(axis=1)
	hits = hits[totals != 0]
	totals = totals[totals != 0]
	err = 1 - (hits /totals)
	ste = (err * (1 - err) / (totals-1)) ** 0.5
	
	n = totals.min() # combine means and standard errors as if all classes had the same number of members as the smallest class
	                 # (for means, that's just a flat average of error rates across classes; for standard errors it's the most conservative way to do it)
	d = [{'mean':err[i], 'ste':ste[i], 'n':n} for i in range(len(totals))]
	d = seqste(*d)
	return d['mean'],d['ste']


def class_loss(true=None, predicted=None, confusion_matrix=None):
	"""
	err, se = class_loss(true, predicted)
	err, se = class_loss(confusion_matrix=C)
		where C = confuse(true, predicted)
	
	Actually class_loss() is symmetrical in its input arguments, but the
	order (true, predicted) is the convention established elsewhere,
	e.g. in balanced_loss()
	
	A classification loss function. As in confuse(), each row of <true>
	or <predicted> is a label for one instance or data point.
	"""###
	
	if confusion_matrix is None:
		predicted = numpy.asarray(predicted).flatten()
		if (predicted > numpy.floor(predicted)).any(): predicted = numpy.sign(predicted)
		confusion_matrix,labels = confuse(true=true, predicted=predicted)

	confusion_matrix = numpy.asarray(confusion_matrix, dtype=numpy.float64)
	n = confusion_matrix.sum()
	err = 1 - confusion_matrix.trace() / n
	se = (err * (1 - err) / (n-1)) ** 0.5
	return err,se

def performance(self, condition=None, type='predictions', labels=None, directory=None):
		# import SigTools
		import os
		import numpy
		
		if directory is None:
			raise Exception("No directory specified")
		directory = os.path.realpath( directory )
		if condition is not None: directory = directory[:-3] + ('%03d' % int(condition))
		files = [os.path.join(directory,x) for x in os.listdir(directory) if x.endswith('_' + type + '.txt')]
		arrays = [numpy.array(eval('\n'.join(open(x).readlines()))) for x in files] 
		confmat = 0
		if labels is None: labels = {'predictions':[1,2], 'responses':[1,2,3]}[type]
		for f,a in zip(files,arrays):
			print(f)
			if len(a) == 0 or numpy.all(a[:,1]==0):
				print("no entries")
			else:
				target = a[:,1]
				achieved = a[:,0]
				good = [int(x in labels) for x in achieved]
				achieved *= good
				if False in good: print("%d bad trials" % (len(good)-sum(good)))
				c,x = confuse(a[:,1],a[:,0], labels=[0]+labels)
				pc,stc = class_loss(confusion_matrix=c)
				pb,stb = balanced_loss(confusion_matrix=c)
				print(c)
				print('Overall accuracy of %s = %3.1f%% +/- %3.1f from %d trials' % (type, 100-100*pc, 100*stc, c.sum()))
				print('Balanced accuracy of %s = %3.1f%% +/- %3.1f from %d trials' % (type, 100-100*pb, 100*stb, c.sum()))
				confmat = confmat + c
			print("\n")
		print("collated over %s" % directory)
		c = confmat
		if c is 0:
			print("no relevant entries found")
		else:			
			pc,stc = class_loss(confusion_matrix=c)
			pb,stb = balanced_loss(confusion_matrix=c)
			print(confmat)
			print(' overall accuracy of %s = %3.1f%% +/- %3.1f from %d trials' % (type, 100-100*pc, 100*stc, c.sum()))
			print('balanced accuracy of %s = %3.1f%% +/- %3.1f from %d trials' % (type, 100-100*pb, 100*stb, c.sum()))

if __name__ == '__main__':
	pass