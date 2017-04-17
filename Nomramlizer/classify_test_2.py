import sys
from Vocab import * 
import numpy as np
from LR_sparse import *

from scipy.sparse import *
import datetime
import calendar 
#from features2 import *
from Feature_Extractor3 import *
import pickle
import matplotlib.pyplot as plt

#import csv
#import xlsxwriter
def get_list_of_target_dates(create_date,range_days):
    list_of_days = []
    for diff in xrange(range_days+1):
        target_date_ = create_date+datetime.timedelta(days=(diff))
        target_date = ((target_date_.isoformat()).strip().split("T"))[0]
        list_of_days.append(target_date)

        if(diff != 0):
            target_date_ = create_date-datetime.timedelta(days=(diff))
            target_date = ((target_date_.isoformat()).strip().split("T"))[0]
            list_of_days.append(target_date)
    list_of_days.sort()
    list_of_days.append("None")
    return list_of_days

def Get_Features(tags,tweet,cr_date,target_date,pos_tags):
    fe = Feature_Extractor(tags,tweet,cr_date,target_date,pos_tags)
    #features = fe.Get_Tweet_Features() + fe.Get_Tag_Features() 
    features = fe.Get_Tweet_Features() + fe.Get_Tag_Features() +fe.Get_POS_Features()
    #features = fe.Get_Tag_Features() 
    #features = fe.Get_Tweet_Features()
    #features = fe.Get_POS_Features()

    #print "***************",len(features)
    
    #features = fe.Get_Tweet_Features() 
    #print "***************",len(features)
    #features = fe.Get_Tag_Features()


    return features



class Date_Resolver:
	"""docstring for Date_Resolver"""
	def __init__(self, inputFile, l='1',r_days=10):
		print inputFile
		self.range_days = r_days
		self.N = 0
		self.vocab = Vocab()
		for line in open(inputFile):
			if line=="\n":continue
			line_values  = line.strip().split("\t")
			tags = line_values[0]
			tweet = line_values[1]
			cr_date = line_values[2]
			ev_date = line_values[3]
			pos_tags = line_values[5]
			
			create_date = None
			
			try:
				create_date=datetime.datetime.strptime(cr_date,'%Y-%m-%d')
			except Exception, e:
				print "error in date convert :: create date :: "+str(e)

			if(create_date != None):
				target_date_list = get_list_of_target_dates(create_date,self.range_days)
				for target_date in target_date_list:
					features = Get_Features(tags,tweet,cr_date,target_date,pos_tags)
					for f in features:
						self.vocab.GetID(f[0])
						self.N += 1


		self.vocab.Lock()
		self.X_matrix = lil_matrix((self.N, self.vocab.GetVocabSize()))
		self.Y = np.zeros(self.N)

		i=0

		for line in open(inputFile):
			if line=="\n":continue
			line_values  = line.strip().split("\t")
			tags = line_values[0]
			tweet = line_values[1]
			cr_date = line_values[2]
			ev_date = line_values[3]
			pos_tags = line_values[5]
			
			create_date = None
			
			try:
				create_date=datetime.datetime.strptime(cr_date,'%Y-%m-%d')
			except Exception, e:
				print "error in date convert :: create date :: "+str(e)

			if(create_date != None):
				target_date_list = get_list_of_target_dates(create_date,self.range_days)
				for target_date in target_date_list:
					features = Get_Features(tags,tweet,cr_date,target_date,pos_tags)
					for f in features:
						self.X_matrix[i,self.vocab.GetID(f[0])-1] = f[1]
						if ev_date == target_date:
							self.Y[i] = 1.0
						else:
							self.Y[i] = -1.0
					i+=1

		print "tocsr"
		self.X_matrix = self.X_matrix.tocsr()
		print "done tocsr"







	def Train(self):
		print "train"
		self.lr = LR(self.X_matrix, self.Y)
		self.lr.Train()










	def Evaluate(self,tags, tweet , cr_date, target_date, pos_tags):
		#print "evaluate"
		features = Get_Features(tags,tweet,cr_date,target_date,pos_tags)
		X = np.zeros(self.vocab.GetVocabSize())
		for f in features:
			if self.vocab.GetID(f[0]) > 0:
				X[self.vocab.GetID(f[0])-1] = f[1]
		Y_pred = self.lr.Predict(X)
		return Y_pred

	def PrintWeights(self, outFile): 
		fOut = open(outFile, 'w') 
		for i in np.argsort(-self.lr.wStar): 
			fOut.write("%s\t%s\n" % (self.vocab.GetWord(i+1), self.lr.wStar[i]))


	def Evaluate_and_Plot_Print(self,testFile,outFile):
		print "Eval & Plot"
		#self.Evalutate()
		Tp =0.0
		Fp=0.0
		Fn = 0.0
		Tn=0.0
		list_date_tuples = [] 
		num_gold_dates = 0


		fOut = open(outFile, 'w') 


		'''workbook = xlsxwriter.Workbook("PR.xlsx")
		worksheet = workbook.add_worksheet()

		row = 0

		worksheet.write(row, 0, 'Precision')
		worksheet.write(row, 1, 'Recall')'''

		for line in open(testFile):
			if line=="\n": continue
			line_values = line.strip().split('\t')
			tags = line_values[0]
			tweet = line_values[1]
			cr_date = line_values[2]
			ev_date = line_values[3]
			pos_tags = line_values[5]
			if(ev_date != "None"): num_gold_dates+=1

			create_date = None

			try:
				create_date=datetime.datetime.strptime(cr_date,'%Y-%m-%d')
			except Exception, e:
				print "error in date convert :: create date :: "+str(e)
			if(create_date == None): continue
			
			target_date_list = get_list_of_target_dates(create_date,self.range_days)
			y_max = -1
			predicted_date = "None"
			correct_prediction = False

			for target_date in target_date_list:
				y = self.Evaluate(tags, tweet, cr_date, target_date, pos_tags)
				#print y
				#t = (tweet,cr_date,ev_date, predicted_date,ymax)
				t = (tweet,cr_date,ev_date, target_date,y)
				list_date_tuples.append(t)
				#print y, cr_date, target_date, ev_date
				if(y>y_max): 
					y_max =y
					predicted_date=target_date

			fOut.write("predicted date: %s \t y_max: %s \t tweet: %s \t ev_date: %s \t cr_date: %s\n\n" % (predicted_date, y_max,tweet,ev_date,cr_date))
			if(predicted_date == ev_date):
				correct_prediction=True

			'''if y_max > 0.5 and correct_prediction == True:
				Tp +=1
			elif y_max <= 0.5 and correct_prediction == False:
				Tn+=1
			elif y_max <= 0.5 and correct_prediction == True:
				Fn+=1
			elif y_max > 0.5 and correct_prediction == False:
				Fp+=1'''
			if (predicted_date == ev_date and predicted_date != "None"):
				Tp+=1
			elif predicted_date!="None" and predicted_date!=ev_date:
				Fp+=1
			#elif ev_date!="None" and predicted_date!=ev_date:
			#	Fp+=1

		
		Fn = num_gold_dates - Tp
		print "(tp, tn, fn, fp) = ",(Tp, Tn, Fp, Fn)
		P = Tp / (Tp + Fp)
		R = Tp / (Tp + Fn)
		F = ( 2 * P * R ) / (P + R)
		print P,R,F
		'''list_date_tuples.sort(key=lambda x: x[4],reverse=True)
		precesion_list=[]
		recall_list=[]
		tp =0.0
		fp =0.0
		fn = 0.0
		#print list_date_tuples

		for t in list_date_tuples:
			event_date = t[2]
			predicted_date =  t[3]
			#print event_date,predicted_date
			if (predicted_date == event_date and predicted_date != "None"):
				tp+=1
			elif predicted_date!="None" and predicted_date!=event_date:
				fp+=1
			fn = num_gold_dates - tp

			if(tp+fp>0):
				p=tp/(float(tp+fp))
				r =tp/(float(tp+fn))
				precesion_list.append(p)
				recall_list.append(r)

		#print precesion_list
		#print recall_list'''
		'''row = 0
		for i in range(len(precesion_list)):
			row+=1
			worksheet.write(row, 0, precesion_list[i])
			worksheet.write(row, 1, recall_list[i])


		plt.plot(recall_list,precesion_list)
		plt.show()'''






if __name__ == '__main__':
	print "main"
	train_file = "data/by_week_mod_5_2_pos/op_small"
	#test_file = "data/1st_50_pos/op"
	test_file = "data/1st_200/4th_50"
	weight_output_file = "weight_wo_pos"
	#test_output_file = "op_wo_pos"
	test_output_file = "outputs/150/2nd_50"
	DR = Date_Resolver(train_file)
	DR.Train()
	DR.Evaluate_and_Plot_Print(test_file,test_output_file)
	#DR.PrintWeights(weight_output_file)
