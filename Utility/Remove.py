#from nltk.corpus import stopwords
import re, string

#cachedStopWords = stopwords.words("english")


cachedStopWords = [ 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',  'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't']

class Removal:
	"""remove stopwords and punctuation"""
	def __init__(self, input_str):
		self.input_string = input_str
		#print self.input_string

	'''def update_string(self,input_str):
	    self.input_string=input_str
	'''
	def remove_punctuation(self):
	    text = re.sub('[%s]' % re.escape(string.punctuation), '', self.input_string)
	    self.input_string = text
	    return text

	def remove_stop_words(self):
	    text = ' '.join([word for word in self.input_string.split() if word not in cachedStopWords])
	    self.input_string = text
	    return text

   
'''
rmvl = Removal("the a :: joi , . ; adfadsfads")
s = rmvl.remove_stop_words()
print s
#s = "string. With. Punctuation?" # Sample string 
#out = re.sub('[%s]' % re.escape(string.punctuation), '', s)
s = rmvl.remove_punctuation()
print s
p = "a  the quick cunning bwero dfsad; '[]"
rmvl = Removal(p)
s = rmvl.remove_stop_words()
print s
#s = "string. With. Punctuation?" # Sample string 
#out = re.sub('[%s]' % re.escape(string.punctuation), '', s)
s = rmvl.remove_punctuation()
print s


'''
