import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

input_file=open("RORC-compound.txt","r")
output_file=open("RORC-compound-Morgan","a")

for line in input_file:
	line=line.strip().split("\t")
	smiles=line[0]
	id=line[1]
	if Chem.MolFromSmiles(smiles):
		mol=Chem.MolFromSmiles(smiles)
		fps=AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
		fingerprints=fps.ToBitString()
		i=0
		while i<1024:
			output_file.write(fingerprints[i]+',')
			i=i+1
		output_file.write(id+'\n')	
		
input_file.close()
output_file.close()


