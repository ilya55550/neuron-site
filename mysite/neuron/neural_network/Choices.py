choices_loss = [
    ('Probabilistic losses', (
        ('binary_crossentropy', 'binary_crossentropy'),
        ('categorical_crossentropy', 'categorical_crossentropy'),
        ('sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'),
        ('poisson', 'poisson'),
        ('kl_divergence', 'kl_divergence'),
    )
     ),
    ('Regression losses', (
        ('mean_squared_error', 'mean_squared_error'),
        ('mean_absolute_error', 'mean_absolute_error'),
        ('mean_absolute_percentage_error', 'mean_absolute_percentage_error'),
        ('mean_squared_logarithmic_error', 'mean_squared_logarithmic_error'),
        ('cosine_similarity', 'cosine_similarity'),
        ('huber_loss', 'huber_loss'),
        ('log_cosh', 'log_cosh'),
    )
     ),
    ('Hinge losses for "maximum-margin" classification', (
        ('hinge', 'hinge'),
        ('squared_hinge', 'squared_hinge'),
        ('categorical_hinge', 'categorical_hinge'),
    )
     ),
]

choices_optimizer = (
    ('SGD', 'SGD'),
    ('RMSprop', 'RMSprop'),
    ('Adam', 'Adam'),
    ('Adadelta', 'Adadelta'),
    ('Adadelta', 'Adadelta'),
    ('Adamax', 'Adamax'),
    ('Nadam', 'Nadam'),
    ('Ftrl', 'Ftrl'),
)
