import unittest


class TestDataLoader(unittest.TestCase):

    def get_dataloader(self):
        if self.data_origin == 'zhang':
            from ncem.data import DataLoaderZhang as DataLoader
            radius = 100

        self.data = DataLoader(
            data_path=self.data_path,
            radius=radius,
        )


class TestDataLoaderZang(TestDataLoader, unittest.TestCase):
    data_path = '/Users/anna.schaar/phd/datasets/zhang/'
    data_origin = 'zhang'

    def test_data_types(self):
        self.get_dataloader()

if __name__ == '__main__':
    unittest.main()
