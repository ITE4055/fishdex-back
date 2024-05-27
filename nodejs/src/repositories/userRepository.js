const pool = require("../config/database");

exports.findAll = async () => {
  const [rows] = await pool.query("SELECT * FROM Users");
  return rows;
};

exports.create = async (userData) => {
  const { userName, userCode } = userData;
  const [result] = await pool.query(
    "INSERT INTO Users (userName, userCode) VALUES (?, ?)",
    [userName, userCode]
  );
  return {
    id: result.insertId,
    userName,
    userCode,
  };
};
