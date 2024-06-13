const userRepository = require("../repositories/userRepository");
const jwt = require("jsonwebtoken");
require("dotenv").config();

// exports.getAllUsers = async () => {
//   return await userRepository.findAll();
// };

exports.login = async (username, usercode) => {
  const user = await userRepository.findByUsercode(usercode);
  let token;

  if (user.length > 0) {
    token = jwt.sign({ usercode }, process.env.SECRET_KEY, { expiresIn: "1h" });
    return { message: "로그인 완료", token };
  }

  await userRepository.register(username, usercode);
  token = jwt.sign({ usercode }, process.env.SECRET_KEY, { expiresIn: "1h" });
  return { message: "회원가입 완료", token };
};

exports.getUserByUserCode = async (usercode) => {
  const user = await userRepository.findByUsercode(usercode);
  if (user.length === 0) {
    throw new Error("회원번호가 존재하지 않습니다.");
  }
  return user[0];
};
