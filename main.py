import os
from os import path
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from python_speech_features import delta
import pickle
from sklearn import mixture
from sklearn.preprocessing import Normalizer
import pandas as pd
import numpy as np
import librosa as lr

class SpeechProcessing:
	class IO:
		def read_wav(self, path): 
			sr, signal = wav.read(path)
			return sr, signal
	
	class FeatsProcessor: 
		def generate_mfcc(self, sr, signal):
			mfccs = mfcc(signal, samplerate=sr, winlen=0.025, winstep=0.01, numcep=13, nfilt=24, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=np.hamming)
			d_mfccs = delta(mfccs, 2)
			dd_mfccs = delta(d_mfccs, 2) 
			mfccs_d_dd = np.hstack([mfccs, d_mfccs, dd_mfccs])
			return mfccs_d_dd
	
	class SOM: 
		def generate_som_labels(self, feats, model_path): 
			norm = Normalizer().fit(feats)
			norm_data = norm.transform(feats)
			som_model =  self.load_model(model_path)
			results = self.predict_labels(som_model, norm_data)
			df = pd.DataFrame(columns=['FrameID','ClusterID'])
			for i in range(len(results)): 
				cluster_id = np.nonzero(results[i])[0][0]
				df=df.append({'FrameID':i,'ClusterID':cluster_id},ignore_index=True)
			return df
			
		def load_model(self,model_path):
			with open(model_path,'rb') as f: 
				model = pickle.load(f)
				return model
		def predict_labels(self, model, data): 
			return model.predict(data) 

	class LM:
		def reduce_frame_label(self,cluster_seq): 
			grammar_list=[]
			seq = cluster_seq.split(',')
			repeat_count = 0
			for i in range(len(seq)-1):
				curr_id = seq[i]
				next_id = seq[i+1]
				if(str(curr_id)!=str(next_id)):
					gram = "{iden}|".format(iden=str(curr_id))
					grammar_list.append(gram)
			grammar_list.append(seq[len(seq)-1])
			return ''.join(grammar_list)
		
		def getFrameDuration(self, start_fr, end_fr, sr): 
			start_dur= 0.00
			end_dur = 0.00
			#print("Sampling rate = ",sr)
			fr_st = int(start_fr)
			fr_end = int(end_fr)
			time_dur = lr.frames_to_time([fr_st,fr_end], sr= sr, hop_length=int(0.01*sr), n_fft=int(0.25*sr))
			if(len(time_dur)==2): 
				start_dur= time_dur[0]
				end_dur = time_dur[1]
			return start_dur, end_dur
			 
		def generate_pseudo_label(self, cluster_arr, base_path, fname):
			sr = 16000 # default sampling rate - For other value change here
			result_file_path= path.join(base_path, "Label","{}.lbl".format(fname))
			repeat_count = 0
			start_frame = 0
			end_frame = 0
			with open(result_file_path,'a') as fout:
				head_str="{},{},{},{},{},{} \n".format("filename","st_fr","end_fr","st_time","end_time","label")
				for i in range(len(cluster_arr)-1): 
					curr_id = cluster_arr[i]
					next_id = cluster_arr[i+1]
					if(curr_id!=next_id):
						if(repeat_count >0):
							end_frame = start_frame + (repeat_count)
							st_time, end_time = self.getFrameDuration(start_frame, end_frame, sr)
							lbl_str="{},{},{},{},{},{} \n".format(fname,start_frame,end_frame,st_time,end_time,str(curr_id))
							#print(lbl_str)
							fout.write(lbl_str)
							repeat_count = 0
							start_frame = end_frame+1
						else:
							end_frame = start_frame 
							st_time, end_time = self.getFrameDuration(start_frame, end_frame, sr)
							#print(lbl_str)
							lbl_str="{},{},{},{},{},{} \n".format(fname,start_frame,end_frame,st_time,end_time,str(curr_id))
							#print(lbl_str)
							fout.write(lbl_str)
							start_frame = end_frame+1
							
					else:
						#end_frame +=1 
						repeat_count +=1
			print("Label expored successfully...")
			return result_file_path
		
		def matrix(self,a, b, match_score=1, gap_cost=0):
			H = np.zeros((len(a) + 1, len(b) + 1), np.int)
			for i in range(1, H.shape[0]): 
				for j in range(1, H.shape[1]): 
					#match = (H[i-1,j-1]+match_score if a[i-1]==b[j-1] else 0)
					match = (match_score if a[i-1]==b[j-1] else 0) 
					H[i,j] = match
			return H 
		
		def traceback(self,H, b, match_score):
			r, c = H.shape
			match_coll=[]
			for rid in range(r-1,0,-1):
				match=[]
				maxi = np.max(H[rid, :])
				cid = np.where(H[rid, :]==maxi)[0][0]
				match.append("{}".format(b[cid-1]))
				H[rid,cid] =1
				rid -=1
				cid -=1
				while(H[rid,cid]>=match_score): 
					match.append("{}".format(b[cid-1]))
					H[rid,cid] =1
					rid -=1
					cid -=1
				
				match.reverse()
				match_coll.append("|".join(match))
				
			return match_coll
				
		def sim_tracer(self, a, b, match_score=1, gap_cost=0):
			a = a.split("|")
			b = b.split("|")
			H = self.matrix(a, b, match_score, gap_cost)
			H1 = H
			match_coll = self.traceback(H, b, match_score)
			return H1, match_coll
		
		
