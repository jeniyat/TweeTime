#!/usr/bin/python


import datetime
import calendar
from Utility.Remove import *
from collections import Counter
import string 

def tag_form_date(target_date,create_date):
    
    tags = list()
    if(target_date=="None"):
        tags.append("NA")
    else:
        dat = target_date[8:]
        tags.append(dat)
        tg_date =None
        try:
            tg_date=datetime.datetime.strptime(target_date,'%Y-%m-%d')
        except Exception, e:
            print "error in date convert :: target date :: "+str(e)

        if(tg_date != None):
            tags.append((tg_date.strftime("%b")).lower())
            tags.append((tg_date.strftime("%a")).lower())

            cr_date = None
            try:
                cr_date=datetime.datetime.strptime(create_date,'%Y-%m-%d')
            except Exception, e:
                print "error in date convert :: create date :: "+str(e)
            if(cr_date != None):
                if(tg_date < cr_date):
                    tags.append("past")
                elif(tg_date > cr_date):
                    tags.append("future")
                elif(tg_date == cr_date):
                    tags.append("present")

            
    return tags

def if_tommorow_yesterday(target_date,create_date):
    tag = ""
    diff = -1
    if(target_date=="None"):
        tag = "NA"
        diff = -1
    else:
        
        tg_date =None
        try:
            tg_date=datetime.datetime.strptime(target_date,'%Y-%m-%d')
        except Exception, e:
            print "error in date convert :: target date :: "+str(e)

        if(tg_date != None):
            cr_date = None
            try:
                cr_date=datetime.datetime.strptime(create_date,'%Y-%m-%d')
            except Exception, e:
                print "error in date convert :: create date :: "+str(e)
            if(cr_date != None):
                if(tg_date < cr_date):
                    if((cr_date - tg_date).days==1):
                    	tag = "yesterday"
                    	diff = 1
                elif (tg_date > cr_date):
        			if((tg_date - cr_date).days==1):
        				tag = "tommorow"
        				diff = 1

    return (tag,diff)



def diff_of_dates(target_date,create_date):
    tag = ""
    diff = -1
    if(target_date=="None"):
        tag="NA"
        diff = -1
    else:
        
        tg_date =None
        try:
            tg_date=datetime.datetime.strptime(target_date,'%Y-%m-%d')
        except Exception, e:
            print "error in date convert :: target date :: "+str(e)

        if(tg_date != None):
            cr_date = None
            try:
                cr_date=datetime.datetime.strptime(create_date,'%Y-%m-%d')
            except Exception, e:
                print "error in date convert :: create date :: "+str(e)
            if(cr_date != None):
                if(tg_date < cr_date):
                    diff = (cr_date - tg_date).days
                    tag = "prev_"+str(diff)
                    diff = 1
                elif(tg_date > cr_date):
                    diff = (tg_date - cr_date).days
                    tag= "next_"+str(diff)
                    diff = 1
                elif(tg_date == cr_date):
                    tag="now"
                    diff = 1

    return (tag,diff)


class Feature_Extractor:
    def __init__(self,  tg,twt, cr_dat, trgt_dat,pos_tags=" "):
        #print "gu kha"
        self.tweet = twt
        self.tag = tg
        self.create_date = cr_dat
        self.target_date = trgt_dat
        self.pos_tags = pos_tags
        
    def Get_POS_Features(self):
        features = list()
        #print self.tweet
        #print self.pos_tags
        pos_list = self.pos_tags.split(" ")


        
        #print word_list
        
        pos_list = [''.join(c for c in s if c not in string.punctuation) for s in pos_list]
        pos_list = [s for s in pos_list if s]

        word_list = self.tweet.split(" ")
        word_list = [''.join(c for c in s if c not in string.punctuation) for s in word_list]
        word_list = [s for s in word_list if s]

        #print word_list
        #print len(word_list)
        #print len(pos_list)
        #if (len(word_list)!=len(pos_list)):return []

        length = min(len(word_list), len(pos_list))
        for i in range(length):
            word = word_list[i]
            pos = pos_list[i]
            tag = word+"_"+pos
            count = 1
            features.append((tag,count))

        '''for i in range(len(pos_list)-1):
            tag1= pos_list[i]
            tag2 = pos_list[i+1]
            tag=tag1+"_"+tag2
            features.append((tag,1))'''
        
        
        #print pos_list
        
        unique_pos_list = sorted(set([c for c in pos_list if c.startswith("VB")]))
        pos_counter = Counter(pos_list)
        #print pos_counter
        
        #print pos_list
        #print unique_pos_list

        

        '''for v in unique_pos_list:
            features.append((v,1))'''
        '''for pos_tag in pos_counter:
            count = pos_counter[pos_tag]
            features.append((pos_tag,count))'''
        


        
        return features

    def Get_Tag_Features(self):
        features = list()

        tags = self.tag.split(',')
        #print tags

        tags_from_target_date = tag_form_date(self.target_date,self.create_date)

        #print tags_from_target_date
        predicted_tags = []

        for t in tags:
            #print t
            t_list = t.split(':')
            tag_name = t_list[0]
            predicted_tags.append(tag_name)
            tag_count = int(t_list[1])
            tag_score = float(t_list[2])


            if tag_name in tags_from_target_date:
                if(tag_name!="NA"):
                    tg = tag_name+"="+"_count"
                    features.append((tg,tag_count))
                    tg = tag_name+"="+"_score"
                    features.append((tg,tag_score))
                tg = tag_name+"="+"_binary"
                features.append((tg,1))

            else:
                if(tag_name!="NA"):
                    tg=tag_name+"+"+"_count"
                    features.append((tg,tag_count))
                    tg = tag_name+"+"+"_score"
                    features.append((tg,tag_score))
                tg = tag_name+"+"+"_binary"
                features.append((tg,1))

        for t in tags_from_target_date:
            #print t
            if t not in predicted_tags:
                #print t
                tg=t+"-"+"_binary"
                features.append((tg,1))

        #print predicted_tags
        #print features

        diff_days = diff_of_dates(self.target_date,self.create_date)

        if(diff_days[1] != -1):
            features.append(diff_days)

        tommorow_yesterday = if_tommorow_yesterday(self.target_date,self.create_date)
        if(tommorow_yesterday[1] != -1):
            features.append(tommorow_yesterday)



        return features

    
    def Get_Tweet_Features(self):
		features = list()

		rmvl = Removal(self.tweet.lower())
		tweet = rmvl.remove_stop_words()
		tweet = rmvl.remove_punctuation()
		

		predicted_tags=list()

		tags = tag_form_date(self.target_date,self.create_date)


		

		for w in tweet.split(' '):
			if(w != '' or w!= ' '):
				for t in tags:
					f_name = w+"_"+t
					#print f_name
					#f_val = word_counter[w]
					features.append((f_name,1))

		return features





		

if __name__ == "__main__":
	#print "gu kha"
	fe = Feature_Extractor('30:1:50.687881503947544,NA:22:0' ,'@shaneylaney maybe we should go to the Nordstrom Anniversary Sale next Monday and get dinner at Cheesecake ? Just a suggestion !! @katpaidas', '2011-07-18',  '2011-07-15','USR RB PRP MD VB TO DT NNP NNP NNP JJ NNP CC VB NN IN NNP . RB DT NN . USR')


	#f= fe.Get_Tag_Features()
	f2 = fe.Get_POS_Features()
	#print f
	print f2

	#print f+f2
	