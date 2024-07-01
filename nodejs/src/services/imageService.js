const imageRepository = require("../repositories/imageRepository");

exports.uploadImage = async (
  usercode,
  url,
  category,
  weight,
  length,
  location
) => {
  await imageRepository.uploadImage(
    usercode,
    url,
    category,
    weight,
    length,
    location
  );

  return { message: "이미지 등록 완료" };
};

exports.getImagesByCategory = async (usercode, category) => {
  const results = await imageRepository.findImagesByCategory(
    usercode,
    category
  );
  if (results.length === 0) {
    throw new Error("결과값이 없습니다. 도감을 채워주세요.");
  }
  return results;
};

exports.getCountByUsercode = async (usercode) => {
  const count = await imageRepository.countImagesByUsercode(usercode);
  console.log("service" + count);
  return count;
};
