import pickle


def save_file(file_full_name, file_content):
    with open(file_full_name, 'wb') as handle:
        pickle.dump(file_content, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(file_full_name):
    with open(file_full_name, 'rb') as handle:
        file_content = pickle.load(handle)
    return file_content
