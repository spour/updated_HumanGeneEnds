import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
print("imports successful")
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, help="joblib of model you want to use for training", required=True)
parser.add_argument("-i", "--infile", type=str, help="list of infile paths of vector of pwm scores for whole genes", required=True)
parser.add_argument("-out", "--outfile", type = str, help="scored out file location", required=True)
#parser.add_argument("-t", "--thresh", type=float, help="threshold for accuracy prediction", default=0)
args = parser.parse_args()
print(args.model, args.infile)

model = joblib.load(args.model)
model_name = Path(args.model).stem
pd.set_option("display.precision", 6)
#infile =  pd.read_csv(args.infile, sep = '\t', index_col = 0)
#infile_name = Path(args.infile).stem
print("opening file")
with open(args.infile) as file1:
#    lines = file1.readlines()
#    lines = [line.rstrip() for line in lines]
     for line in file1:
         line = line.rstrip()
         f = pd.read_csv(line, sep = '\t', index_col = 0, header = None)
         output = np.column_stack([f.index, model.predict_proba(f.values)[:,1]])
         DF=pd.DataFrame(output)
         #DF[['Gene','allele']] = DF[0].str.split('_',expand=True)
         DF[['Gene','allele', "loc"]] = DF[0].str.split('_',expand=True)
         #DF.columns = ["full_name", "pred_score", "Gene", "allele"]
         DF.columns = ["full_name", "pred_score", "Gene", "allele", "loc"]
         df_new = pd.concat([DF['Gene'], DF['full_name'], DF['pred_score']], axis = 1)
         for region, df_region in df_new.groupby('Gene'):
             out = args.outfile+'/' +region
             df_region.to_csv(out, sep = '\t')
             print("done:", region)


#f=pd.concat([pd.read_csv(f, sep = '\t', index_col = 0) for f in lines])
#output = np.column_stack([f.index, model.predict_proba(f.values)[:,1]])
#DF=pd.DataFrame(output)
#DF[['Gene','allele']] = DF[0].str.split('_',expand=True)
#DF.columns = ["full_name", "pred_score", "Gene", "allele"]
#df_new = pd.concat([DF['Gene'], DF['full_name'], DF['pred_score']], axis = 1)
#for region, df_region in df_new.groupby('Gene'):
#    out = args.outfile+'/' +region
#    df_region.to_csv(out, sep = '\t')
#    print("done:", region)



#output = np.column_stack([infile.index, model.predict_proba(infile.values)[:,1]])
#DF = pd.DataFrame(output)
#you would probably want the outfile to have the name of the gene in it
#DF.to_csv(args.outfile, header = False, sep = '\t', compression='gzip')
#print("DONE")

