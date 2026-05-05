import { IsEmail, IsNotEmpty, MaxLength } from 'class-validator';

/* 只接受 name、email 就好了，id 是自动生成的，createdAt、updatedAt 也会自动更新值 */

export class CreateUserDto {
  @IsNotEmpty()
  @MaxLength(50)
  name: string;

  @IsNotEmpty()
  @IsEmail()
  @MaxLength(50)
  email: string;
}
