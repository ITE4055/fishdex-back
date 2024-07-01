const userService = require("../services/userService");

exports.login = async (req, res) => {
  const { username, usercode } = req.body;

  if (!username || !usercode) {
    return res
      .status(400)
      .json({ message: "사용자명과 회원번호를 입력해주세요." });
  }

  try {
    const result = await userService.login(username, usercode);
    return res.status(200).json(result);
  } catch (error) {
    console.error(error);
    return res.status(500).json({ message: "Internal server error" });
  }
};

exports.getUserByUsercode = async (req, res) => {
  const { usercode } = req.body;

  if (!usercode) {
    return res.status(400).json({ message: "회원번호를 입력해주세요." });
  }

  try {
    const result = await userService.getUserByUserCode(usercode);
    return res.status(200).json(result);
  } catch (error) {
    console.error(error);
    return res.status(500).json({ message: "Internal server error" });
  }
};

// exports.getAllUsers = async (req, res, next) => {
//   try {
//     const users = await userService.getAllUsers();
//     res.json(users);
//   } catch (error) {
//     next(error);
//   }
// };

// exports.login = async (req, res, next) => {
//   try {
//     const user = await userService.createUser(req.body);
//     res.status(201).json(user);
//   } catch (error) {
//     next(error);
//   }
// };
