import pandas as pd
import os
import wget
import zipfile


class Pamap2Handler():
    def __init__(self, cache_dir, out = print):
        """
        cache_dir: directory where to download and extract dataset
        out: callable which receives log strings, defaults to print
        """
        self.url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip" 
        self.cache_dir = cache_dir
        self.log = out

        self.protocol_dir = os.path.join(cache_dir, "PAMAP2_Dataset/Protocol/")
        self.optional_dir = os.path.join(cache_dir, "PAMAP2_Dataset/Optional/")

        self.all_cols = [
            'timestamp',
            'activity_id',
            'heart_rate', 'h_temperature', 
            'h_xacc16','h_yacc16', 'h_zacc16',
            'h_xacc6', 'h_yacc6', 'h_zacc6', 'h_xgyr', 'h_ygyr', 'h_zgyr',
            'h_xmag', 'h_ymag', 'h_zmag',
            'ho1','ho2','ho3','ho4',
            'c_temperature',
            'c_xacc16', 'c_yacc16', 'c_zacc16', 'c_xacc6', 'c_yacc6', 'c_zacc6',
            'c_xgyr', 'c_ygyr', 'c_zgyr',
            'c_xmag', 'c_ymag', 'c_zmag',
            'co1','co2','co3','co4',
            'a_temperature',
            'a_xacc16', 'a_yacc16', 'a_zacc16', 'a_xacc6',
            'a_yacc6', 'a_zacc6', 'a_xgyr', 'a_ygyr', 'a_zgyr',
            'a_xmag', 'a_ymag','a_zmag',
            'ao1','ao2','ao3','ao4',
        ]

        self.valid_cols = [
            'timestamp',
            'activity_id',
            'heart_rate', 'h_temperature', 
            'h_xacc16','h_yacc16', 'h_zacc16',
            'h_xacc6', 'h_yacc6', 'h_zacc6', 'h_xgyr', 'h_ygyr', 'h_zgyr',
            'h_xmag', 'h_ymag', 'h_zmag',
            'c_temperature',
            'c_xacc16', 'c_yacc16', 'c_zacc16', 'c_xacc6', 'c_yacc6', 'c_zacc6',
            'c_xgyr', 'c_ygyr', 'c_zgyr',
            'c_xmag', 'c_ymag', 'c_zmag',
            'a_temperature',
            'a_xacc16', 'a_yacc16', 'a_zacc16', 'a_xacc6',
            'a_yacc6', 'a_zacc6', 'a_xgyr', 'a_ygyr', 'a_zgyr',
            'a_xmag', 'a_ymag','a_zmag'
        ]

    def download(self, dest_dir=None):
        if dest_dir  is None:
            dest_dir = self.cache_dir
        url = self.url
        dataset_dir = "PAMAP2_Dataset"
        zip_filename = f"{dataset_dir}.zip"
        path_zip = os.path.join(dest_dir, zip_filename)
        if not dataset_dir in os.listdir(dest_dir):
            if not zip_filename in os.listdir(dest_dir):
                self.log("download")
                wget.download(url, out=dest_dir)
            else:
                self.log("already downloaded")   
            with zipfile.ZipFile(path_zip, "r") as zip_ref:
                zip_ref.extractall(dest_dir)
    
    
    def optional_subject_path(self, subject:int):
        return os.path.join(self.optional_dir, f"subject10{subject}_opt.dat")

    def protocol_subject_path(self, subject:int):
        return os.path.join(self.protocol_dir, f"subject10{subject}.dat")

    def extract_df(self, fpath):
        self.download()
        df = pd.read_table(fpath, sep="\s+", names=self.all_cols)
        return df[self.valid_cols]
    
    def get_protocol_subject(self, subject: int):
        path = self.protocol_subject_path(subject)
        return self.extract_df(path)

    def get_optional_subject(self, subject: int):
        path = self.optional_subject_path(subject)
        return self.extract_df(path)


def cross_validation_split(dfs, transformer_tr, transformer_val, transformer_ts, idx_val, idx_ts):
    idxs_tr = list(range(len(dfs)))

    for idx in [idx_val, idx_ts]:
        o  =idxs_tr.index(idx)
        idxs_tr = [*idxs_tr[:o], *idxs_tr[o+1:]]
    
    d_tr = [transformer_tr.transform(dfs[i]) for i in idxs_tr]
    d_val = [transformer_val.transform(dfs[idx_val])]
    d_ts = [transformer_ts.transform(dfs[idx_ts])]
    return d_tr, d_val, d_ts