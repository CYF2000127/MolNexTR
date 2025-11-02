""" Set of common abbreviations in molecular images"""
from typing import List
import re


ORGANIC_SET = {'B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I'}

RGROUP_SYMBOLS = ['R', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12',
                  'Ra', 'Rb', 'Rc', 'Rd', 'Rf', 'X', 'Y', 'Z', 'Q', 'A', 'E', 'Ar', 'Ar1', 'Ar2', 'Ari', 'Ar3', 'Ar4','Ar5','Ar6','Ar7',"R'", 
                  '1*', '2*','3*', '4*','5*', '6*','7*', '8*','9*', '10*','11*', '12*','[a*]', '[b*]','[c*]', '[d*]',"EWG",'Nu']

PLACEHOLDER_ATOMS = ["Lv", "Lu", "Nd", "Yb", "At", "Fm", "Er"]

class Substitution(object):
    '''Define common substitutions for chemical shorthand'''
    def __init__(self, abbrvs, smarts, smiles, probability):
        assert type(abbrvs) is list
        self.abbrvs = abbrvs
        self.smarts = smarts
        self.smiles = smiles
        self.probability = probability

SUBSTITUTIONS: List[Substitution] = [
    Substitution(['NO2', 'O2N'], '[N+](=O)[O-]', "[N+](=O)[O-]", 0.5),
    Substitution(['N2'], '[N+]=[N-]', "[N+]=[N-]", 0),
    Substitution(['CHO', 'OHC'], '[CH1](=O)', "[CH1](=O)", 0.5),
    Substitution(['CO2Et', 'COOEt','EtO2C'], 'C(=O)[OH0;D2][CH2;D2][CH3]', "[C](=O)OCC", 0.5),

    Substitution(['OAc'], '[OH0;X2]C(=O)[CH3]', "[O]C(=O)C", 0.7),
    Substitution(['NHAc'], '[NH1;D2]C(=O)[CH3]', "[NH]C(=O)C", 0.7),
    Substitution(['Ac'], 'C(=O)[CH3]', "[C](=O)C", 0.1),

    Substitution(['OBz'], '[OH0;D2]C(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[O]C(=O)c1ccccc1", 0.7),  # Benzoyl
    Substitution(['Bz'], 'C(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[C](=O)c1ccccc1", 0.2),  # Benzoyl

    Substitution(['OBn'], '[OH0;D2][CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[O]Cc1ccccc1", 0.7),  # Benzyl
    Substitution(['Bn'], '[CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[CH2]c1ccccc1", 0.2),  # Benzyl

    Substitution(['NHBoc'], '[NH1;D2]C(=O)OC([CH3])([CH3])[CH3]', "[NH1]C(=O)OC(C)(C)C", 0.6),
    Substitution(['NBoc'], '[NH0;D3]C(=O)OC([CH3])([CH3])[CH3]', "[NH1]C(=O)OC(C)(C)C", 0.6),
    Substitution(['Boc'], 'C(=O)OC([CH3])([CH3])[CH3]', "[C](=O)OC(C)(C)C", 0.2),

    


    Substitution(['Cbm'], 'C(=O)[NH2;D1]', "[C](=O)N", 0.2),
    Substitution(['Cbz'], 'C(=O)OC[cH]1[cH][cH][cH1][cH][cH]1', "[C](=O)OCc1ccccc1", 0.4),
    Substitution(['Cy'], '[CH1;X3]1[CH2][CH2][CH2][CH2][CH2]1', "[CH1]1CCCCC1", 0.3),
    Substitution(['OCy'], '[O]C1CCCCC1', "[O]C1CCCCC1", 0.5),  # Phenyl
    Substitution(['Fmoc'], 'C(=O)O[CH2][CH1]1c([cH1][cH1][cH1][cH1]2)c2c3c1[cH1][cH1][cH1][cH1]3',
                 "[C](=O)OCC1c(cccc2)c2c3c1cccc3", 0.6),
    Substitution(['Mes'], '[cH0]1c([CH3])cc([CH3])cc([CH3])1', "[c]1c(C)cc(C)cc(C)1", 0.5),
    Substitution(['OMs'], '[OH0;D2]S(=O)(=O)[CH3]', "[O]S(=O)(=O)C", 0.7),
    Substitution(['Ms'], 'S(=O)(=O)[CH3]', "[S](=O)(=O)C", 0.2),
    Substitution(['Ph'], '[cH0]1[cH][cH][cH1][cH][cH]1', "[c]1ccccc1", 0.5),
    Substitution(['PMB'], '[CH2;D2][cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', "[CH2]c1ccc(OC)cc1", 0.2),
    Substitution(['Py'], '[cH0]1[n;+0][cH1][cH1][cH1][cH1]1', "[c]1ncccc1", 0.1),
    Substitution(['SEM'], '[CH2;D2][CH2][Si]([CH3])([CH3])[CH3]', "[CH2]CSi(C)(C)C", 0.2),
    Substitution(['Suc'], 'C(=O)[CH2][CH2]C(=O)[OH]', "[C](=O)CCC(=O)O", 0.2),
    Substitution(['TBS'], '[Si]([CH3])([CH3])C([CH3])([CH3])[CH3]', "[Si](C)(C)C(C)(C)C", 0.5),
    Substitution(['OTBS'], 'O[Si](C)(C)C(C)(C)CC', "O[Si](C)(C)C(C)(C)CC", 0.5),
    Substitution(['TBZ'], 'C(=S)[cH]1[cH][cH][cH1][cH][cH]1', "[C](=S)c1ccccc1", 0.2),
    Substitution(['OTf'], '[OH0;D2]S(=O)(=O)C(F)(F)F', "[O]S(=O)(=O)C(F)(F)F", 0.7),
    Substitution(['Tf'], 'S(=O)(=O)C(F)(F)F', "[S](=O)(=O)C(F)(F)F", 0.2),
    Substitution(['TFA'], 'C(=O)C(F)(F)F', "[C](=O)C(F)(F)F", 0.3),
    Substitution(['TMS',"SiR2","SiR23"], '[Si]([CH3])([CH3])[CH3]', "[Si](C)(C)C", 0.5),
    Substitution(['Ts'], 'S(=O)(=O)c1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', "[S](=O)(=O)c1ccc(C)cc1", 0.6),  # Tos
    Substitution(['OTMS',"OSiR2","OSiR23"], 'O[Si](C)(C)C', "O[Si](C)(C)C", 0.5),
    Substitution(['OPO(OEt)2'], '[O]P(=O)(OCC)OCC', "[O]P(=O)(OCC)OCC", 0.2),  # Tos
    Substitution(['OPO(OMe)2'], '[O]P(=O)(OC)OC', "[O]P(=O)(OC)OC", 0.2),
    Substitution(['TBDPSO','OTBDPS'], '[O][Si](C(C)(C)C)(c1ccccc1)c1ccccc1', '[O][Si](C(C)(C)C)(c1ccccc1)c1ccccc1', 0.2),  # Tos
    Substitution(['SO2Ph'], '[S](=O)(=O)c1ccccc1', '[S](=O)(=O)c1ccccc1', 0.5),
    Substitution(['SO2Me'], '[S](=O)(=O)[CH3]', '[S](=O)(=O)C', 0.5),
    Substitution(['SO2Et'], '[S](=O)(=O)[CH2;D2][CH3]', '[S](=O)(=O)CC', 0.5),
    Substitution(['SO2iPr'], '[S](=O)(=O)[CH1;D3]([CH3])[CH3]', '[S](=O)(=O)C(C)C', 0.5),
    Substitution(['SO2tBu'], '[S](=O)(=O)[CH0]([CH3])([CH3])[CH3]', '[S](=O)(=O)C(C)(C)C', 0.5),
    Substitution(['Piv'], '[C](=O)C(C)(C)C', '[C](=O)C(C)(C)C', 0.5),
    Substitution(['PivO','OPiv'], '[O]C(=O)C(C)(C)C', "[O]C(=O)C(C)(C)C", 0.5),  # Phenyl



    # Alkyl chains

    Substitution(['OMe', 'MeO'], '[OH0;D2][CH3;D1]', "[O]C", 0.3),
    Substitution(['OAr'], '[O](*)', '[O](*)', 0.5),
    Substitution(['SMe', 'MeS'], '[SH0;D2][CH3;D1]', "[S]C", 0.3),
    Substitution(['NMe', 'MeN'], '[N;X3][CH3;D1]', "[NH]C", 0.3),
    Substitution(['Me'], '[CH3;D1]', "[CH3]", 0.1),
    Substitution(['OEt', 'EtO'], '[OH0;D2][CH2;D2][CH3]', "[O]CC", 0.5),
    Substitution(['Et', 'C2H5'], '[CH2;D2][CH3]', "[CH2]C", 0.3),
    Substitution(['Pr', 'nPr', 'n-Pr'], '[CH2;D2][CH2;D2][CH3]', "[CH2]CC", 0.3),
    Substitution(['Bu', 'nBu', 'n-Bu'], '[CH2;D2][CH2;D2][CH2;D2][CH3]', "[CH2]CCC", 0.3),
    Substitution(['OPh', 'OPh'], '[O]c1ccccc1', "[O]c1ccccc1", 0.2),

    # Branched
    Substitution(['iPr', 'i-Pr'], '[CH1;D3]([CH3])[CH3]', "[CH1](C)C", 0.2),
    Substitution(['iPrO', 'i-PrO','OiPr'], '[OH0;D2][CH1;D3]([CH3])[CH3]', "[O]C(C)C", 0.2),
    Substitution(['iBu', 'i-Bu'], '[CH2;D2][CH1;D3]([CH3])[CH3]', "[CH2]C(C)C", 0.2),
    Substitution(['OiBu'], '[OH0;D2][CH2;D2][CH1;D3]([CH3])[CH3]', "[O]CC(C)C", 0.2),
    Substitution(['OtBu'], '[OH0;D2][CH0]([CH3])([CH3])[CH3]', "[O]C(C)(C)C", 0.6),
    Substitution(['tBu', 't-Bu'], '[CH0]([CH3])([CH3])[CH3]', "[C](C)(C)C", 0.3),
    Substitution(['CO2Me', 'MeO2C'], '[C](=O)OC', "[C](=O)OC", 0.3),
    Substitution(['MeO2CO', 'OCO2Me'], '[O]C(=O)OC', "[O]C(=O)OC", 0.3),
    Substitution(['ONa', 'NaO'], '[O-].[Na+]', "[O-].[Na+]", 0.3),

    # Other shorthands (MIGHT NOT WANT ALL OF THESE)
    Substitution(['CF3', 'F3C'], '[CH0;D4](F)(F)F', "[C](F)(F)F", 0.5),
    Substitution(['NCF3', 'F3CN'], '[N;X3][CH0;D4](F)(F)F', "[NH]C(F)(F)F", 0.5),
    Substitution(['OCF3', 'F3CO'], '[OH0;X2][CH0;D4](F)(F)F', "[O]C(F)(F)F", 0.5),
    Substitution(['CCl3'], '[CH0;D4](Cl)(Cl)Cl', "[C](Cl)(Cl)Cl", 0.5),
    Substitution(['CO2H', 'HO2C', 'COOH'], 'C(=O)[OH]', "[C](=O)O", 0.5),  # COOH
    Substitution(['CN', 'NC'], 'C#[ND1]', "[C]#N", 0.5),
    Substitution(['OCH3', 'H3CO','CH3O'], '[OH0;D2][CH3]', "[O]C", 0.4),
    Substitution(['SO3H'], 'S(=O)(=O)[OH]', "[S](=O)(=O)O", 0.4),


    ###MOLNEXTR
    Substitution(['C5H17','C5H11','C5H14'], 'CCCCC', "CCCCC", 0.0),
    Substitution(['C4H9','C4H10'], 'CCCC', "CCCC", 0.0),
    Substitution(['C3H7','C3H8'], 'CCC', "CCC", 0.0),
    Substitution(['C2H5','C2H6'], 'CC', "CC", 0.0),
    Substitution(['C11H23','C17H23'], 'CCCCCCCCCCC', "CCCCCCCCCCC", 0.0),
    Substitution(['Alyl','Allyl'], 'C=CC', "C=CC", 0.0),
    Substitution(['OAll','OAlI'], 'OCC=C', "OCC=C", 0.0),

    Substitution(['N3'], 'N=[N+]=[N-]', "N=[N+]=[N-]", 0.2),
    Substitution(['N2+'], 'N#[N+]', "N#[N+]", 0),
    Substitution(['N2'], '[N+]=[N-]', "[N+]=[N-]", 0),
    Substitution(['Tos','Tcs'], 'S(=O)(=O)c1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', "[S](=O)(=O)c1ccc(C)cc1", 0),  # Tos
    Substitution(['OTBDMS'], '[OH0;D2][Si](C)(C)C(C)(C)C', "[O][Si](C)(C)C(C)(C)C", 0),  # TBDMS
    Substitution(['SP'], 'S[P]', "S[P]", 0), # Sulfenyl Phosphide
    Substitution(['CH3O'], '[OH0;D2][CH3]', "[O]C", 0),
    Substitution(['OCN'], 'N=C=O', "N=C=O", 0),
    Substitution(['SO2NH2'], 'S(N)(=O)=O', "S(N)(=O)=O", 0),
    Substitution(['NHCOtBu'], 'NC(=O)C(C)(C)C', "NC(=O)C(C)(C)C", 0),
    Substitution(['SPh'], 'Sc1ccccc1', "Sc1ccccc1", 0),
    Substitution(['EtOH'], '[CH2;D2][CH3;D1][OH0;D2]', "[CH2]CO", 0),  # Ethanol
    Substitution(['TBA'], '[CH3;D1][C;D4]([CH3;D1])([CH3;D1])[CH3;D1]', "[CH3]C(C)(C)C", 0),  # Tert-Butyl alcohol
    Substitution(['DMF'], 'CN(C)C=O', "CN(C)C=O", 0),  # Dimethylformamide
    Substitution(['DMSO'], 'CS(=O)C', "CS(=O)C", 0),  # Dimethyl sulfoxide
    Substitution(['THF'], 'C1CCCO1', "C1CCCO1", 0),  # Tetrahydrofuran
#Substitution(['C19H15'], 'CC1=CC=CC=C1C2=CC=CC=C2C3=CC=CC=C3', 'CC1=CC=CC=C1C2=CC=CC=C2C3=CC=CC=C3', 0.5),# Trityl
    ### complex substituents
    Substitution(['-ClO4'], "Cl([O-])(=O)(=O)=O", "Cl([O-])(=O)(=O)=O", 0),  
    Substitution(['-OTf'], "O=S(=O)([O-])C(F)(F)F", "O=S(=O)([O-])C(F)(F)F", 0),  
    Substitution(['-BF4','BF4'], 'F[B-](F)(F)F', "F[B-](F)(F)F", 0),  



    Substitution(['Ac'], 'CC(=O)', "CC(=O)", 0.2),
    Substitution(['Me3SiO'], '[O][Si](C)(C)C', "[O][Si](C)(C)C", 0.2),
    Substitution(['CO2CH2Bn'], '[C](=O)O[CH2]c1ccccc1', "[C](=O)O[CH2]c1ccccc1", 0.2),
    Substitution(['NMe'], '[N]C', "[N]C", 0.2),
    Substitution(['TIPS'], '[Si](C(C)C)(C(C)C)C(C)C', '[Si](C(C)C)(C(C)C)C(C)C', 0.2),
    Substitution(['TlPS'], 'C', 'C', 0.2),
    Substitution(['C6F5'], ' c1c(F)c(F)c(F)c(F)c1(F)', ' [c]1c(F)c(F)c(F)c(F)c1(F)', 0.2),
    Substitution(['OC6Cl5'], '[O]c1c(Cl)c(Cl)c(Cl)c(Cl)c1(Cl)', '[O]c1c(Cl)c(Cl)c(Cl)c(Cl)c1(Cl)', 0.2),
    Substitution(['nC5H11','C5H11'], 'CCCCC', 'CCCCC', 0.2),
    Substitution(['nC4H9','C4H9'], 'CCCC', 'CCCC', 0.2),
    Substitution(['pTol','Tol'], '[c]1ccc(C)cc1', '[c]1ccc(C)cc1', 0.2),
    Substitution(['PMP'], '[C]2=CC=C(OC)C=C2', '[C]2=CC=C(OC)C=C2', 0.2),
    Substitution(['OPMP'], '[O]C2=CC=C(OC)C=C2', '[O]C2=CC=C(OC)C=C2', 0.2),
    Substitution(['NMe2','Me2N'], '[N](C)C', '[N](C)C', 0.2),
    Substitution(['C(O)NMe2'], '[C](=O)N(C)C', '[C](=O)N(C)C', 0.2),
    Substitution(['C(O)Et'], '[C](=O)CC', '[C](=O)CC', 0.2),
    Substitution(['CHPh2',"CH(Ph)2"], '[CH](c1ccccc1)c1ccccc1', '[CH](c1ccccc1)c1ccccc1', 0.2),



    Substitution(['4-BrC6H4','BrC6H4'], 'c1ccc(Br)cc1', "c1ccc(Br)cc1", 0.4),
    Substitution(['2-BrC6H4'], 'c1c(Br)cccc1', "c1c(Br)cccc1", 0.4),
    Substitution(['3-BrC6H4'], 'c1cc(Br)ccc1', "c1cc(Br)ccc1", 0.4),
    Substitution(['CF3C6H3', '3,5-CF3C6H3','(CF3)C6H3'], '[C](F)(F)F', "[C](F)(F)F", 0.4),
    Substitution(['4-CO2MeC6H4'], 'C(=O)Oc1ccc(C)cc1', "[c]1ccc(C(=O)OC)cc1", 0.5),
    Substitution(['3-CO2MeC6H4'], 'C(=O)Oc1cc(C)ccc1', "[c]1cc(C(=O)OC)ccc1", 0.5),
    Substitution(['1-Napth','Napdh','17Napdh'], 'c1ccc2ccccc2c1', "c1ccc2ccccc2c1",0.5),
    Substitution(['2-MeC6H4'], 'c1c(C)cccc1', "c1c(C)cccc1", 0.5),
    Substitution(['3-MeC6H4'], 'c1cc(C)ccc1', "c1cc(C)ccc1", 0.5),
    Substitution(['4-MeC6H4','MeC6H4','AeC6H4','4MeC6H4'], 'c1ccc(C)cc1', "c1ccc(C)cc1", 0.5),
    Substitution(['4-OMeC6H4','OMeC6H4'], 'c1ccc(OC)cc1', "c1ccc(OC)cc1", 0.5),
    Substitution(['3-OMeC6H4','3OMeC6H4'], 'c1cc(OC)ccc1', "c1cc(OC)ccc1", 0.5),
    Substitution(['4-MeOC6H4','MeOC6H4','p-MeO-C6H4','4-MeO'], 'c1ccc(CO)cc1', "c1ccc(CO)cc1", 0.5),
    Substitution(['2-MeOC6H4','C6H4OMe-2'], 'c1c(CO)cccc1', "c1c(CO)cccc1", 0.5),
    Substitution(['2-ClC6H4'], 'c1c(Cl)cccc1', "c1c(Cl)cccc1", 0.2),
    Substitution(['4-ClC6H4','C6H4Cl-4','ClC6H4'], 'c1ccc(Cl)cc1', "c1ccc(Cl)cc1", 0.5),
    Substitution(['4-FC6H4'], 'c1ccc(F)cc1', "c1ccc(F)cc1", 0.5),  
    Substitution(['4-CF3C6H4','A4CF3C6H4','4CF3C6H4'], 'c1ccc(cc1)C(F)(F)F', "c1ccc(cc1)C(F)(F)F", 0.5),
    Substitution(['4-NO2C6H4','NO2C6H4','PNP','4NO2C6H4'], '[c]1ccc(cc1)[N+](=O)[O-]', "[c]1ccc(cc1)[N+](=O)[O-]", 0.5),
    Substitution(['2-thienyl'], '[c]1[s]ccc1', "[c]1[s]ccc1", 0.5),
    Substitution(['2-furyl','Z'], '[c]1occc1', "[c]1occc1", 0.5),
    Substitution(['2-pyridyl'], 'c1ncccc1', "c1ncccc1", 0.5),
    Substitution(['3-pyridyl'], 'c1ccncc1', "c1ccncc1", 0.5),
    Substitution(['2,4-Cl2C6H3','2, 4-Cl2C6H3','Cl2C6H3'], 'c1c(Cl)cc(Cl)cc1', "c1c(Cl)cc(Cl)cc1", 0.5),

    Substitution(['[CF3]2C6H3', '3,5-[CF3]2C6H3','3,5-(CF3)2C6H3'], '[c]1cc(C(F)(F)F)cc(C(F)(F)F)c1', "[c]1cc(C(F)(F)F)cc(C(F)(F)F)c1", 0.4),

   ###NEW
    Substitution(['B(OH)2','B(0H)2'],'B(O)O','B(O)O',0.5),
    Substitution(['CF2H','HF2C','F2C'],'C(F)(F)','C(F)(F)',0.5),
    Substitution(['SCF3','ScF3','SO3F'],'SC(F)(F)F','SC(F)(F)F',0.5),
    Substitution(['F3'],'(F)(F)F','(F)(F)F',0.5),
    Substitution(['AgSe','AgScF3'],'[Ag+].[S-]C(F)(F)F','[Ag+].[S-]C(F)(F)F',0.5),
    Substitution(['Me3Si'],'[Si](C)(C)C','[Si](C)(C)C',0.5),
    Substitution(['(H)'],'H','H',0.5),
    Substitution(['CN'],'C#N','C#N',0.5),
    Substitution(['SN'],'SN','SN',0.5),

    Substitution(['N((SO2Ph))2'],'N(S(=O)(=O)c1ccccc1)S(=O)(=O)c1ccccc1','N(S(=O)(=O)c1ccccc1)S(=O)(=O)c1ccccc1',0.5),
    Substitution(['NHTs'],'NS(=O)(=O)c1ccc(cc1)C','NS(=O)(=O)c1ccc(cc1)C',0.5),
    Substitution(['OCOCH3'],'OC(=O)C','OC(=O)C',0.5),
    Substitution(['CO2tBu', 'CO2tBu'], 'C(=O)OC(C)(C)C', "C(=O)OC(C)(C)C", 0.3),#这个替换好像不太对
    Substitution(['ScH3','SCH3'],'SC','SC',0.5),
    Substitution(['n-Pent'],'CCCCC','CCCCC',0.5),
    Substitution(['H3','H'],'','',0.5),

]

ABBREVIATIONS = {abbrv: sub for sub in SUBSTITUTIONS for abbrv in sub.abbrvs}

VALENCES = {
    "H": [1], "Li": [1], "Be": [2], "B": [3], "C": [4], "N": [3, 5], "O": [2], "F": [1],
    "Na": [1], "Mg": [2], "Al": [3], "Si": [4], "P": [5, 3], "S": [6, 2, 4], "Cl": [1], "K": [1], "Ca": [2],
    "Br": [1], "I": [1]
}

ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]

COLORS = {
    u'c': '0.0,0.75,0.75', u'b': '0.0,0.0,1.0', u'g': '0.0,0.5,0.0', u'y': '0.75,0.75,0',
    u'k': '0.0,0.0,0.0', u'r': '1.0,0.0,0.0', u'm': '0.75,0,0.75'
}

# tokens of condensed formula
FORMULA_REGEX = re.compile(
    '(?:' + '|'.join([re.escape(k) for k in ABBREVIATIONS.keys()]) + '|R[0-9]*|[A-Z][a-z]+|[A-Z]|[0-9]+|\(|\))')
