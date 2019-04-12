import omniglot_loader


loader = omniglot_loader.OmniglotLoader(dataset_path='./Omniglot Dataset/', 
    use_augmentation=False, batch_size=10)

loader.split_train_datasets()
X, y = loader.get_train_batch()

print(X)
print(y)

