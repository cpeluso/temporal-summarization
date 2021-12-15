default_params = dict(
  LEARNING_RATE       = 1e-5,
  IS_BACKBONE_TRAINED = True,
  EPOCHS              = 2,
  MAX_GRAD_NORM       = 10,
  N_LOGGING_STEPS     = 100,
  MAX_LEN             = 512,
  TRAIN_SIZE          = 0.8,
  TEST_SIZE           = 0.5, # of TRAIN_SIZE
  TRAIN_BATCH_SIZE    = 2,
  VALID_BATCH_SIZE    = 1,
  TEST_BATCH_SIZE     = 1,
  TRAIN_SHUFFLE       = True,
  VALID_SHUFFLE       = False,
  TEST_SHUFFLE        = False,
  NUM_WORKERS         = 0
)