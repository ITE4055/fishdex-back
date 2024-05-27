const userRepository = require('../repositories/userRepository');

exports.getAllUsers = async () => {
  return await userRepository.findAll();
};

exports.createUser = async (userData) => {
  return await userRepository.create(userData);
};