const pool = require("../config/database");

exports.uploadImage = async (usercode, url, category) => {
  const [result] = await pool.query(
    "INSERT INTO images (usercode, url, category) VALUES (?, ?, ?)",
    [usercode, url, category]
  );
  return result;
};

exports.findImagesByCategory = async (usercode, category) => {
  const [results] = await pool.query(
    "SELECT url FROM images WHERE usercode = ? AND category = ?",
    [usercode, category]
  );
  return results;
};
