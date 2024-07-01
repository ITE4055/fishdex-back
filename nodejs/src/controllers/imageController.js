const imageService = require("../services/imageService");

exports.uploadImage = async (req, res) => {
  const { usercode, url, category, weight, length, location } = req.body;

  try {
    const result = await imageService.uploadImage(
      usercode,
      url,
      category,
      weight,
      length,
      location
    );
    console.log("controller" + result);
    return res.status(200).json(result);
  } catch (error) {
    console.log(error);
    return res.status(500).json({ message: "Internal server error" });
  }
};

exports.getImagesByCategory = async (req, res) => {
  const { usercode } = req.body;
  const category = req.params.category;

  try {
    const image = await imageService.getImagesByCategory(usercode, category);
    return res.status(200).json(image);
  } catch (error) {
    if (error.message === "결과값이 없습니다. 도감을 채워주세요.") {
      return res.status(401).json({ message: error.message });
    }
    return res.status(500).json({ message: "Internal server error" });
  }
};

exports.getCountByUsercode = async (req, res) => {
  const { usercode } = req.body;

  try {
    const count = await imageService.getCountByUsercode(usercode);
    console.log("controller" + count);
    return res.status(200).json(count);
  } catch (error) {
    console.log(error);
    return res.status(500).json({ message: "Internal server error" });
  }
};
