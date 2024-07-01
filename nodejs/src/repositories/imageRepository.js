const pool = require("../config/database");

exports.uploadImage = async (
  usercode,
  url,
  category,
  weight,
  length,
  location
) => {
  const [result] = await pool.query(
    "INSERT INTO images (usercode, url, category, weight, length, location) VALUES (?, ?, ?, ?, ?, ?)",
    [usercode, url, category, weight, length, location]
  );
  console.log("repository" + result);
  return result;
};

exports.findImagesByCategory = async (usercode, category) => {
  const [results] = await pool.query(
    "SELECT * FROM images WHERE usercode = ? AND category = ?",
    [usercode, category]
  );
  return results;
};

exports.countImagesByUsercode = async (usercode) => {
  const [count] = await pool.query(
    "SELECT category, COUNT(*) AS count FROM images WHERE usercode = ? GROUP BY category",
    [usercode]
  );
  console.log("repository" + count);
  return count;
};
