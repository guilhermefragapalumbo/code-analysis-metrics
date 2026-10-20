
import { ApiService } from '../api.service';
import { HouseData } from '../house-data';
import { Component, ViewChild } from '@angular/core';
import { NgForm } from '@angular/forms';


@Component({
  selector: 'app-componente-home',
  templateUrl: './componente-home.component.html'
})
export class ComponenteHomeComponent  {

  data:any;
  obj: any;
  houseData: HouseData = {
    numBathrooms: 0,
    numRooms: 0,
    location: '',
    size: 0,
    outdoorSpace: 0,
    price:0
  };

  locations = [    { 'id': '8', 'name': 'Puerto Rico' },    { 'id': '11', 'name': 'Virgin Islands' },    { 'id': '3', 'name': 'Massachusetts' },    { 'id': '0', 'name': 'Connecticut' },    { 'id': '4', 'name': 'New Hampshire' },    { 'id': '10', 'name': 'Vermont' },    { 'id': '5', 'name': 'New Jersey' },    { 'id': '6', 'name': 'New York' },    { 'id': '9', 'name': 'Rhode Island' },    { 'id': '2', 'name': 'Maine' },    { 'id': '7', 'name': 'Pennsylvania' },    { 'id': '1', 'name': 'Delaware' }]


  constructor(private apiService: ApiService) {}
  @ViewChild('myForm') form!: NgForm;

  // ngOnInit() {
  //   this.apiService.getData().subscribe(data => {
  //     this.data = data;
  //   });
  // }

getPrice() {
    this.apiService.getPriceService(this.houseData).subscribe(
      response => {
        //this.obj = response;
        this.houseData.price = response;
        
      },
      error => console.error(error)
    );
  }
}
