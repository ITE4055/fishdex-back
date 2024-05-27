const userRepository = require("../repositories/userRepository");
const jwt = require("jsonwebtoken");
require("dotenv").config();

exports.getAllUsers = async () => {
  return await userRepository.findAll();
};

exports.createUser = async (userData) => {
  return await userRepository.create(userData);
};

exports.loginOrRegister = async (username, usercode) => {
  const users = await userRepository.findByUsercode(usercode);
  let token;

  if (users.length > 0) {
    token = jwt.sign({ usercode }, process.env.SECRET_KEY, { expiresIn: "1h" });
    return { message: "로그인 완료", token };
  }

  await userRepository.createUser(username, usercode);
  token = jwt.sign({ usercode }, process.env.SECRET_KEY, { expiresIn: "1h" });
  return { message: "회원 등록 완료", token };
};
