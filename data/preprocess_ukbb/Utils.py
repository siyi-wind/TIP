import os

import torch
import numpy as np
import pandas as pd


def check_or_save(obj, path, index=None, header=None):
    if isinstance(obj, pd.DataFrame):
        if index is None or header is None:
            raise ValueError(
                "Index and header must be specified for saving a dataframe"
            )
        if os.path.exists(path):
            if not header:
                saved_df = pd.read_csv(path, header=None)
            else:
                saved_df = pd.read_csv(path)
            naked_df = saved_df.reset_index(drop=True)
            naked_df.columns = range(naked_df.shape[1])
            naked_obj = obj.reset_index(drop=not index)
            naked_obj.columns = range(naked_obj.shape[1])
            if naked_df.round(6).equals(naked_obj.round(6)):
                return
            else:
                diff = naked_df.round(6) == naked_obj.round(6)
                diff[naked_df.isnull()] = naked_df.isnull() & naked_obj.isnull()
                assert diff.all().all(), "Dataframe is not the same as saved dataframe"
        else:
            obj.to_csv(path, index=index, header=header)
    else:
        if os.path.exists(path):
            saved_obj = torch.load(path)
            if isinstance(obj, list):
                for i in range(len(obj)):
                    check_array_equality(obj[i], saved_obj[i])
            else:
                check_array_equality(obj, saved_obj)
        else:
            print(f"Saving to {path}")
            torch.save(obj, path)


def check_array_equality(ob1, ob2):
    if torch.is_tensor(ob1) or isinstance(ob1, np.ndarray):
        assert (ob2 == ob1).all()
    else:
        assert ob2 == ob1
