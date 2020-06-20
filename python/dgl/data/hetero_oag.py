import os
import numpy as np
import pickle

from .utils import download, extract_archive, get_download_dir, _get_dgl_url
from ..utils import retry_method_with_fix
from .. import backend as F

class OAG:
    def __init__(self, name=''):
        """Initialize the dataset.

        Paramters
        ---------
        name : str
            ('CS', 'med', '').
        """
        assert name in ['CS', 'med', '']
        self._dir = get_download_dir()
        self._name = name
        self._zip_file_path = '{}/OAG_{}.zip'.format(self._dir, self._name)
        self._load()

    def _download(self):
        _url = 'dataset/OAG/OAG_{}.zip'.format(self._name)
        download(_get_dgl_url(_url), overwrite=False, path=self._zip_file_path)
        extract_archive(self._zip_file_path,
                        '{}/OAG_{}'.format(self._dir, self._name))

    @retry_method_with_fix(_download)
    def _load(self):
        """Loads input data.
        """
        print('Loading G...')
        dir_path = os.path.join(self._dir, "OAG_{}".format(self._name))
        dir_path = os.path.join(dir_path, "OAG_{}".format(self._name))
        self.graph = pickle.load(open(dir_path + '/graph.pkl', 'rb'))
        emb = np.load(dir_path + '/affil_emb.npy')
        self.graph.nodes['affiliation'].data['emb'] = F.tensor(emb)
        emb = np.load(dir_path + '/field_emb.npy')
        self.graph.nodes['field'].data['emb'] = F.tensor(emb)
        field = np.load(dir_path + '/field_field.npy')
        self.graph.nodes['field'].data['field'] = F.tensor(field)
        emb = np.load(dir_path + '/paper_emb.npy')
        self.graph.nodes['paper'].data['emb'] = F.tensor(emb)
        time = np.load(dir_path + '/paper_time.npy')
        self.graph.nodes['paper'].data['year'] = F.tensor(time)
        emb = np.load(dir_path + '/venue_emb.npy')
        self.graph.nodes['venue'].data['emb'] = F.tensor(emb)
