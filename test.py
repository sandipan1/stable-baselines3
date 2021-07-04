




  tmp_path = "/tmp/sb3_log/"
  # set up logger
  new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])


    model = A2C("MlpPolicy", "CartPole-v1", verbose=1)
  # Set new logger
  model.set_logger(new_logger)