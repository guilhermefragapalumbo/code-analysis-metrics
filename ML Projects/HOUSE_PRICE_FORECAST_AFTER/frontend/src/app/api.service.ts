import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { HouseData } from './house-data';



@Injectable({
  providedIn: 'root'
})
export class ApiService {

private url = 'http://localhost:8081/';

  constructor(private http: HttpClient,) {}

  getData() {
    return this.http.get(this.url);
  }

  sendData(data: any) {

    return this.http.post(this.url, data);
  }

  getPriceService(_house:HouseData) {
    console.log(_house)
    return this.http.post(this.url, _house);
  }
}


