import os
import json
import numpy as np
from tqdm import tqdm
from paths import raw_data_dir
from paths import npz_data_dir
from constants import all_configurations
from spektral.data import Dataset, Graph

class Act2Act(Dataset):
    """
    Dataset class that organizes refined Act2Act 3DSkeleton data into graphs. Inherits from spektral.data.Dataset

    ...

    Attributes
    ----------
    actions : list
        List of actions/labels to import
    configuration : str, optional
        One of three camera configurations in Act2Act
    expected_extension : str, optional
        Stores expected extension of raw input data. Used to check wheter a directory contains any unwanted files
    y_type : str, optional
        Desired form of data labelling. Label can be defined as string or scalar value
    reshaping_mode : str, optional
        Tells class how to behave when padding node feature matrix, default value does not apply any padding to nfm
    padding_value : int, optional
        Tells class which number to use when applying padding to the node feature matrix
    num_of_feats: int, optional
        Desired number of features held in one node of a single graph. Needs to be passed when user wants to apply padding

    Methods
    -------
    read()
        Generates list of graphs based on downloaded .npz data
    download()
        Gets called when a directory specified in self.path doesn't exist. Turns raw data into organized .npz data used by read() method
    generate_file_list(fpath)
        Generates list of files in specified in fpath raw data folder and filters it based on configuration attribute
    validate_file(file_name, expected_ext)
        Checks if file given by file_name has expected extension specified in expected_ect, and if file actually exists
    read_joint(file_name)
        Loads .JSON file specified by file_name and returns it as a python object
    get_node_feature_matrix(joint_data)
        Returns node feature matrix used to describe number of nodes in graph and number of features in one node
    get_adjacency_matrix(type)
        Returns adjacency matrix usef to describe which nodes of a graph are connected with each other
    get_label(file_name, label_type)
        Returns a label of a given label_type based on label stated in file_name
    new_file_name(fpath, file_name)
        Generates a new path and extension for file in file_name
    reshape_node_feature_matrix(self, nf_matrix, resh_mode, feats_target, padding_val)
        Applies padding to nfm according to reshaping_mode and num_of_feats

    Properties
    ----------
    path : str
        Overrides default path attribute with desired path
    """

    def __init__(self, actions, configuration="C001", expected_extension="joint", y_type="scalar", reshaping_mode="none", padding_value=0, num_of_feats=1000, **kwargs):
        """
        Parameters
        ----------
        actions : list
            List of actions/labels to import
        configuration : str, optional
            One of three camera configurations in Act2Act
        expected_extension : str, optional
            Stores expected extension of raw input data. Used to check wheter a directory contains any unwanted files
        y_type : str, optional
            Desired form of data labelling. Label can be defined as string or scalar value
        reshaping_mode : str, optional
            Tells class how to behave when padding node feature matrix, default value does not apply any padding to nfm
        padding_value : int, optional
            Tells class which number to use when applying padding to the node feature matrix
        num_of_feats: int, optional
            Desired number of features held in one node of a single graph. Needs to be passed when user wants to apply padding
        """
        self.actions = actions
        self.n_classes = len(actions)
        self.configuration = configuration
        self.expected_extension = expected_extension
        self.y_type = y_type
        self.reshaping_mode = reshaping_mode
        self.padding_value = padding_value
        self.num_of_feats = num_of_feats
        super().__init__(**kwargs)

    def read(self):
        """ Generates a list of graphs based by node_feature_matrix, adjacency matrix an label provided in .npz file.
        Firstly function generates list of files to be proccessed and concatenates file path to each item in list. 
        Then, .npz file is loaded using np.load and converted to spektral Graph.

        Additionaly reshapes nfm matrix to a desired shape and normalizes it so that all of its elements are in range <-1:1>.
        
        Returns
        -------
        output 
            A list of graphs made from .npz data in self.path directory
        """

        output = []
        npz_list = [f for f in os.listdir(self.path)]

        for i in range(len(npz_list)):
            npz_path = os.path.join(self.path, npz_list[i])
            npz_list[i] = npz_path

        npz_list_filtered = []
        for npz_file in npz_list:
            npz_name = npz_file.split("\\")[-1][4:8]
            if npz_name in self.actions:
                npz_list_filtered.append(npz_file)

        progress_bar = tqdm(total=len(npz_list_filtered), desc="Proccessing npz data: ")

        for npz_file_filtered in npz_list_filtered:
            data = np.load(npz_file_filtered, allow_pickle=True)

            Y = np.zeros((self.n_classes, ))
            action_name = str(npz_file_filtered.split("\\")[-1][4:8])
            action_idx = self.actions.index(action_name)
            Y[action_idx] = 1

            X = self.reshape_node_feature_matrix(data['x'], self.reshaping_mode, self.num_of_feats, self.padding_value)
            if type(X) is type(None):
                continue

            # Normalization in range -1 1
            up, low = 1, -1
            X += -(np.min(X))
            X /= np.max(X) / (up - low)
            X += low

            output.append(Graph(x=X, a=data['a'], y=Y))
            progress_bar.update(1)
        
        return output

    def download(self):
        """ Gets called only when self.path directory doesn't exist. Function generates list of files to be proccessed into x, a and y matrices of .npz file.
        Function also checks wheter provided file name is of correct type, based on extension.All the data is saved to self.path directory.
        """

        os.mkdir(self.path)

        file_list = self.generate_file_list(raw_data_dir)

        progress_bar = tqdm(total=len(file_list), desc="Proccessing raw data: ")

        for file in file_list:
            self.validate_file(file, self.expected_extension)
            joint_dat = self.read_joint(file)
            x = self.get_node_feature_matrix(joint_dat)
            a = self.get_adjacency_matrix()
            y = self.get_label(file, self.y_type)

            save_pth = self.new_file_name(self.path, file)
            np.savez(save_pth, x=x, a=a, y=y)
            progress_bar.update(1)

    @property
    def path(self):
        """ Overrides default path to which proccessed data is to be saved.

        Returns
        -------
        npz_data_dir
            Path to directory for proccesed from raw data .npz files
        """

        return npz_data_dir
    
    def generate_file_list(self, fpath):
        """ Returns list of files to be proccesed in download() based on provided path to raw data files. Function checks if passed fpath file path is correct, then gets list of all
        element in fpath directory. Then, based with on given configuration, function chooses correct files and puts them in a new, filtered list.

        Parameters
        ----------
        fpath : str
            Path to directory which stores raw data to be filtered and proccessed

        Raises
        ------
        Exception
            If provided fpath path is not a directory

        Returns
        -------
        filtered_list
            List of files to be proccessed in download() method
        """

        if os.path.isdir(fpath):
            print("Provided path is a valid directory")
        else:
            raise Exception("Provided path is not a valid directory")
    
        file_list = [file for file in os.listdir(fpath)]
        print("Successfully generated list of files in a directory")

        filtered_list = []
        for item in file_list:
            if item[:4] == self.configuration:
                filtered_list.append(item)
        else:
            pass

        print(f"Successfully filtered list of files based on given configuration. Length of filtered list: {len(filtered_list)}")

        return filtered_list
    
    def validate_file(self, file_name, expected_ext):
        """ Checks if file passed in file_name is actually a file and if its type is corresponding to expected_extension passed in class constructor

        Parameters
        ----------
        file_name : str
            Name of the file to be validated
        expected_ext : str
            Name of the extension to be validated

        Raises
        ------
        Exception : File does not exist in data folder
            Gets called if file passed in file_name is actually in raw data folder
        Exception : Wrong file format
            Gets called if file is of wrong type, not match the expected_extension parameter
        """

        file_extension = file_name.split(".")[-1]
        file_path = os.path.join(raw_data_dir, file_name)

        if os.path.isfile(file_path):
            pass
        else:
            raise Exception("File does not exist in data folder")
    
        if file_extension == expected_ext:
            pass
        else:
            raise Exception("Wrong file format")
        
    def read_joint(self, file_name):
        """ Reads .joint raw data file as a JSON file and converts it to a python object

        Parameters
        ----------
        file_name : str
            File name of a file to be imported 

        Returns
        -------
        joint_data
            Python object converted from JSON-type .joint file
        """

        joint_path = os.path.join(raw_data_dir, file_name)

        with open(joint_path) as file:
            data = file.read()
            joint_data = json.loads(data)

        return joint_data
    
    def get_node_feature_matrix(self, joint_data):
        """ Generates node_feature_matrix of size (n_nodes, n_features). There are n_nodes nodes in a single nfm, and each node contains x,y,z joint data from all the frames in a file
        Function also handles missing frames, ommiting them from node feature matrix. Resulting matrix might have fewer node_features than expected.

        Parameters
        ----------
        joint_data : python obj

        
        Excepts
        -------
        TypeError
            Skips to the next iteration if TypeError arises. Some files are missing frames, which can result in a index error, where instead of 'joints' property function wants to 
            access 'Null' property of joint_data

        Returns
        -------
        x
            Node feature matrix 
        """

        n_nodes = 25
        n_frames = len(joint_data)

        nfm = []
    
        for joint in range(n_nodes):
            node = []
            for frame in range(n_frames):
                try:
                    x, y, z = joint_data[frame][1]['joints'][joint]['x'], joint_data[frame][1]['joints'][joint]['y'], joint_data[frame][1]['joints'][joint]['z']
                    node.extend([x, y, z])
                except TypeError:
                    pass

            nfm.append(node)

        x = np.array(nfm)

        return x
        
    def get_adjacency_matrix(self):
        """ Generates adjacency matrix which tells spektral which nodes in graph are connected. For each node index stored in R, function matches it to its corresponding entry in C list.
        Then, a matrix of shape (n_joints, n_joints) is created, connected joints are set to 1, rest entries of a matrix are kept at 0,

        Returns
        -------
        a
            Adjacency matrix as a dense numpy matrix. Be aware that some models might require adjacency matrix in sparse format instead
        """

        n_joints = 25

        R = [i for i in range(n_joints)]
        C = [1, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11]

        a = np.zeros([n_joints, n_joints])
        for joint in range(len(R)):
            i, j = R[joint], C[joint]
            a[i][j] = 1
    
        return a
        
    def get_label(self, file_name, label_type):
        """ Generates label of a graph either in scalar or string form. Configuration and action is extracted directly from file name.
        Label in scalar mode is a one-hot label

        Parameters
        ----------
        file_name : str
            Name of the file for which function generates a label
        label_type : str
            Type of returned label - either scalar or a string

        Raises
        ------
        Exception
            Labelling proccess is stopped when passed label_type isn't scalar or string

        Returns
        -------
        label_scalar
            Label as a one-hot labeled vector of length n_classes
        label_string
            Label as a traditional text
        """
        y = np.zeros((self.n_classes, ))

        if label_type in ["scalar","string"]:
            pass
        else:
            raise Exception("Specified type doesn't exist or isn't implemented yet.")
    
        configuration = int(file_name[:4][-1])
        action = file_name[8:12]

        label_scalar = np.array([int(action[-1])])

        label_string = all_configurations[configuration][str(action)]
    
        if label_type == "scalar":
            return label_scalar
        else:
            return label_string
        
    def new_file_name(self, fpath, file_name):
        """ Gets name of a raw data file and converts it to suit new naming convention

        Parameters
        ----------
        fpath : str
            Path to a .npz file storage directory
        file_name : str
            File name to be proccessed by the method

        Returns
        -------
        new_file_name : str
            Path where proccessed data is to be stored, with updated name and type
        """

        cut_name = file_name.split(".")[0][4:]
        new_file_name = os.path.join(fpath, f"{cut_name}.npz")

        return new_file_name
    
    def reshape_node_feature_matrix(self, nf_matrix, resh_mode, feats_target, padding_val):
        """ Method applies padding to nfm. resh_mode=none return the same matrix, upscale mode takes only those matrices, that have less features than feats_target wants, 
        and applies padding to them till they match (n_nodes, feats_target) shape. Mixed mode reshapes all sizes of matrices to match (n_nodes, feats_target) shape.
        Main difference between mixed and upscale mode is that upscale mode discards matrices that have more features than feats_target.
        
        Parameters
        ----------
        nf_matrix : numpy.array
            Node feature matrix to be reshaped
        resh_mode : str
            One of three reshaping modes - 'none', 'upscale', 'mixed'
        feats_target : int
            Target number of node features
        padding_val : int
            nf_matrix is padded with value passed in padding_val'
        
        Returns
        -------
        output_matrix : numpy.array
            Reshaped node feature matrix
        """

        output_matrix = None
        nodes, feats = nf_matrix.shape[0], nf_matrix.shape[1]
        subtracted = feats_target - feats

        if resh_mode == 'none':
            output_matrix = nf_matrix
            return output_matrix
    
        if resh_mode == "upscale": 
            if feats > feats_target:
                return output_matrix
            else:
                if subtracted == 0:
                    output_matrix = nf_matrix
                    return output_matrix
            
                if subtracted > 0:
                    pv_padding = np.ones([nodes, subtracted]) * padding_val
                    output_matrix = np.concatenate((nf_matrix, pv_padding), axis=1)
                    return output_matrix
    
        if resh_mode == "mixed":
            if subtracted == 0:
                output_matrix = nf_matrix
                return output_matrix

            if subtracted < 0:
                output_matrix = nf_matrix[:, :subtracted]
                return output_matrix
            
            if subtracted > 0:
                pv_padding = np.ones([nodes, subtracted]) * padding_val
                output_matrix = np.concatenate((nf_matrix, pv_padding), axis=1)
                return output_matrix
            