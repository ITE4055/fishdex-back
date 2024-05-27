module.exports = (sequelize, DataTypes) => {
  const User = sequelize.define("User", {
    userName: DataTypes.STRING,
    userCode: DataTypes.STRING,
  });
  return User;
};
