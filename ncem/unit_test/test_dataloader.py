import unittest


class TestDataLoader(unittest.TestCase):
    data_origin: str
    data_path: str

    def get_dataloader(self):
        if self.data_origin == "zhang":
            from ncem.data import DataLoaderZhang as DataLoader

            radius = 100
            label_selection = []
        elif self.data_origin == "jarosch":
            from ncem.data import DataLoaderJarosch as DataLoader

            radius = 100
            label_selection = []
        elif self.data_origin == "hartmann":
            from ncem.data import DataLoaderHartmann as DataLoader

            radius = 100
            label_selection = ["Diagnosis", "Age", "Sex"]
        elif self.data_origin == "pascualreguant":
            from ncem.data import DataLoaderPascualReguant as DataLoader

            radius = 100
            label_selection = []
        elif self.data_origin == "schuerch":
            from ncem.data import DataLoaderSchuerch as DataLoader

            radius = 100
            label_selection = ["Group"]
        else:
            from ncem.data import DataLoaderZhang as DataLoader

            radius = 100
            label_selection = []

        self.data = DataLoader(data_path=self.data_path, radius=radius, label_selection=label_selection)


class TestDataLoaderZang(TestDataLoader, unittest.TestCase):
    data_path = "/Users/anna.schaar/phd/datasets/zhang/"
    data_origin = "zhang"

    def test_data_types(self):
        self.get_dataloader()


class TestDataLoaderJarosch(TestDataLoader, unittest.TestCase):
    data_path = "/Users/anna.schaar/phd/datasets/busch/"
    data_origin = "jarosch"

    def test_data_types(self):
        self.get_dataloader()


class TestDataLoaderHartmann(TestDataLoader, unittest.TestCase):
    data_path = "/Users/anna.schaar/phd/datasets/hartmann/"
    data_origin = "hartmann"

    def test_data_types(self):
        self.get_dataloader()


class TestDataLoaderPascualReguant(TestDataLoader, unittest.TestCase):
    data_path = "/Users/anna.schaar/phd/datasets/pascualreguant/"
    data_origin = "pascualreguant"

    def test_data_types(self):
        self.get_dataloader()


class TestDataLoaderSchuerch(TestDataLoader, unittest.TestCase):
    data_path = "/Users/anna.schaar/phd/datasets/schuerch/"
    data_origin = "schuerch"

    def test_data_types(self):
        self.get_dataloader()


if __name__ == "__main__":
    unittest.main()
